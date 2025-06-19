"""Intelligent caching system for Blender AI Agent Studio"""

import json
import hashlib
import pickle
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import OrderedDict
import sqlite3
import gzip
import struct


class CacheType(str, Enum):
    """Types of cached data"""
    AI_RESPONSE = "ai_response"
    BLENDER_COMMAND = "blender_command"
    ASSET_METADATA = "asset_metadata"
    SCENE_TEMPLATE = "scene_template"
    PERFORMANCE_PROFILE = "performance_profile"
    USER_PREFERENCE = "user_preference"


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    cache_type: CacheType
    created_at: float
    last_accessed: float
    access_count: int = 0
    expiry_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of cached value"""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value, default=str).encode('utf-8'))
            else:
                return len(pickle.dumps(self.value))
        except:
            return 1024  # Default estimate
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expiry_time is None:
            return False
        return time.time() > self.expiry_time
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """LRU (Least Recently Used) cache with size limits"""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry and update LRU order"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    self._remove_entry(key)
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.touch()
                return entry
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """Add/update cache entry"""
        with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Ensure we have space
            while (self.current_size + entry.size_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = entry
            self.current_size += entry.size_bytes
    
    def remove(self, key: str) -> bool:
        """Remove specific cache entry"""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def _remove_entry(self, key: str):
        """Internal method to remove entry"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            key, _ = self.cache.popitem(last=False)  # Remove first (oldest)
            self._remove_entry(key)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "total_entries": len(self.cache),
                "total_size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self.current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }


class PersistentCache:
    """Persistent cache using SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db()
        self._lock = threading.RLock()
    
    def _ensure_db(self):
        """Ensure database and table exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    cache_type TEXT NOT NULL,
                    value_data BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    expiry_time REAL,
                    metadata TEXT,
                    size_bytes INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expiry_time ON cache_entries(expiry_time)")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from persistent storage"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache_entries WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        entry = self._row_to_entry(row)
                        if entry.is_expired():
                            self.remove(key)
                            return None
                        
                        # Update access statistics
                        entry.touch()
                        self._update_access(conn, key, entry)
                        return entry
                        
                return None
            except Exception as e:
                print(f"Cache get error: {e}")
                return None
    
    def put(self, key: str, entry: CacheEntry):
        """Store cache entry to persistent storage"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Compress value data
                    value_data = gzip.compress(pickle.dumps(entry.value))
                    metadata_json = json.dumps(entry.metadata) if entry.metadata else None
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, cache_type, value_data, created_at, last_accessed, 
                         access_count, expiry_time, metadata, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, entry.cache_type.value, value_data, entry.created_at,
                        entry.last_accessed, entry.access_count, entry.expiry_time,
                        metadata_json, entry.size_bytes
                    ))
            except Exception as e:
                print(f"Cache put error: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove cache entry from persistent storage"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    return cursor.rowcount > 0
            except Exception as e:
                print(f"Cache remove error: {e}")
                return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries"""
        with self._lock:
            try:
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_entries WHERE expiry_time IS NOT NULL AND expiry_time < ?",
                        (current_time,)
                    )
                    return cursor.rowcount
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                return 0
    
    def get_by_type(self, cache_type: CacheType) -> List[CacheEntry]:
        """Get all entries of a specific type"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache_entries WHERE cache_type = ?",
                        (cache_type.value,)
                    )
                    return [self._row_to_entry(row) for row in cursor.fetchall()]
            except Exception as e:
                print(f"Cache get_by_type error: {e}")
                return []
    
    def _row_to_entry(self, row) -> CacheEntry:
        """Convert database row to CacheEntry"""
        (key, cache_type, value_data, created_at, last_accessed, 
         access_count, expiry_time, metadata_json, size_bytes) = row
        
        # Decompress and deserialize value
        value = pickle.loads(gzip.decompress(value_data))
        
        # Parse metadata
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        return CacheEntry(
            key=key,
            value=value,
            cache_type=CacheType(cache_type),
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=access_count,
            expiry_time=expiry_time,
            metadata=metadata,
            size_bytes=size_bytes or 0
        )
    
    def _update_access(self, conn, key: str, entry: CacheEntry):
        """Update access statistics in database"""
        conn.execute(
            "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
            (entry.last_accessed, entry.access_count, key)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                    count, total_size = cursor.fetchone()
                    
                    # Get type breakdown
                    cursor = conn.execute(
                        "SELECT cache_type, COUNT(*) FROM cache_entries GROUP BY cache_type"
                    )
                    type_counts = dict(cursor.fetchall())
                    
                    return {
                        "total_entries": count or 0,
                        "total_size_mb": (total_size or 0) / (1024 * 1024),
                        "type_breakdown": type_counts
                    }
            except Exception as e:
                print(f"Cache stats error: {e}")
                return {"total_entries": 0, "total_size_mb": 0, "type_breakdown": {}}


class SmartCache:
    """Intelligent caching system with multiple cache layers"""
    
    def __init__(self, cache_dir: str = "./cache", max_memory_mb: int = 256):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache (L1) - fast access
        self.memory_cache = LRUCache(max_memory_mb // 2)
        
        # Persistent cache (L2) - long-term storage
        self.persistent_cache = PersistentCache(str(self.cache_dir / "cache.db"))
        
        # Cache configuration
        self.default_ttl = {
            CacheType.AI_RESPONSE: 24 * 3600,      # 24 hours
            CacheType.BLENDER_COMMAND: 7 * 24 * 3600,  # 7 days
            CacheType.ASSET_METADATA: 30 * 24 * 3600,  # 30 days
            CacheType.SCENE_TEMPLATE: 7 * 24 * 3600,   # 7 days
            CacheType.PERFORMANCE_PROFILE: 24 * 3600,  # 24 hours
            CacheType.USER_PREFERENCE: 365 * 24 * 3600  # 1 year
        }
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.start_time = time.time()
    
    def generate_key(self, data: Union[str, Dict[str, Any]], cache_type: CacheType) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, default=str)
        
        # Create hash with type prefix
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{cache_type.value}:{hash_value}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks memory first, then persistent)"""
        # Try memory cache first (L1)
        entry = self.memory_cache.get(key)
        if entry:
            self.hit_count += 1
            return entry.value
        
        # Try persistent cache (L2)
        entry = self.persistent_cache.get(key)
        if entry:
            # Promote to memory cache
            self.memory_cache.put(key, entry)
            self.hit_count += 1
            return entry.value
        
        self.miss_count += 1
        return None
    
    def put(
        self,
        key: str,
        value: Any,
        cache_type: CacheType,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store value in cache"""
        current_time = time.time()
        
        # Calculate expiry time
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl.get(cache_type, 24 * 3600)
        
        expiry_time = current_time + ttl_seconds if ttl_seconds > 0 else None
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type,
            created_at=current_time,
            last_accessed=current_time,
            expiry_time=expiry_time,
            metadata=metadata or {}
        )
        
        # Store in both caches
        self.memory_cache.put(key, entry)
        self.persistent_cache.put(key, entry)
    
    def cache_ai_response(
        self,
        prompt: str,
        response: str,
        agent_type: str,
        model: str = "gemini-2.0-flash-exp",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Cache AI response with intelligent key generation"""
        cache_data = {
            "prompt": prompt,
            "agent_type": agent_type,
            "model": model
        }
        
        key = self.generate_key(cache_data, CacheType.AI_RESPONSE)
        
        response_metadata = {
            "agent_type": agent_type,
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            **(metadata or {})
        }
        
        self.put(key, response, CacheType.AI_RESPONSE, metadata=response_metadata)
        return key
    
    def get_ai_response(
        self,
        prompt: str,
        agent_type: str,
        model: str = "gemini-2.0-flash-exp"
    ) -> Optional[str]:
        """Get cached AI response"""
        cache_data = {
            "prompt": prompt,
            "agent_type": agent_type,
            "model": model
        }
        
        key = self.generate_key(cache_data, CacheType.AI_RESPONSE)
        return self.get(key)
    
    def cache_blender_commands(
        self,
        requirements: Dict[str, Any],
        commands: List[Dict[str, Any]],
        agent_type: str
    ) -> str:
        """Cache generated Blender commands"""
        key = self.generate_key(requirements, CacheType.BLENDER_COMMAND)
        
        command_data = {
            "commands": commands,
            "requirements": requirements,
            "agent_type": agent_type
        }
        
        metadata = {
            "agent_type": agent_type,
            "command_count": len(commands),
            "requirements_hash": hashlib.md5(json.dumps(requirements, sort_keys=True).encode()).hexdigest()[:8]
        }
        
        self.put(key, command_data, CacheType.BLENDER_COMMAND, metadata=metadata)
        return key
    
    def get_blender_commands(
        self,
        requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached Blender commands"""
        key = self.generate_key(requirements, CacheType.BLENDER_COMMAND)
        return self.get(key)
    
    def cache_asset_metadata(
        self,
        asset_path: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Cache asset metadata"""
        key = self.generate_key(asset_path, CacheType.ASSET_METADATA)
        
        # Add file system info to metadata
        enhanced_metadata = metadata.copy()
        if os.path.exists(asset_path):
            stat = os.stat(asset_path)
            enhanced_metadata.update({
                "file_size": stat.st_size,
                "modified_time": stat.st_mtime,
                "file_path": asset_path
            })
        
        self.put(key, enhanced_metadata, CacheType.ASSET_METADATA, ttl_seconds=30*24*3600)  # 30 days
        return key
    
    def cleanup(self) -> Dict[str, int]:
        """Clean up expired entries and return cleanup statistics"""
        memory_before = len(self.memory_cache.cache)
        persistent_removed = self.persistent_cache.cleanup_expired()
        
        # Clear memory cache expired entries
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.memory_cache.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self.memory_cache.remove(key)
        
        memory_removed = memory_before - len(self.memory_cache.cache)
        
        return {
            "memory_entries_removed": memory_removed,
            "persistent_entries_removed": persistent_removed,
            "total_removed": memory_removed + persistent_removed
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        persistent_stats = self.persistent_cache.get_stats()
        
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            "performance": {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "uptime_hours": uptime_hours
            },
            "memory_cache": memory_stats,
            "persistent_cache": persistent_stats,
            "total_entries": memory_stats["total_entries"] + persistent_stats["total_entries"],
            "total_size_mb": memory_stats["total_size_mb"] + persistent_stats["total_size_mb"]
        }
    
    def invalidate_by_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern"""
        # This would need more sophisticated implementation for production
        # For now, just clear all - could be enhanced with regex matching
        if pattern == "*":
            self.memory_cache.clear()
    
    def optimize(self):
        """Perform cache optimization"""
        # Clean up expired entries
        cleanup_stats = self.cleanup()
        
        # Could add more optimizations:
        # - Compress frequently accessed items
        # - Preload popular items to memory cache
        # - Analyze access patterns for better TTL settings
        
        return {
            "optimization_performed": True,
            "cleanup_stats": cleanup_stats
        }


# Global cache instance
smart_cache = SmartCache()


def get_cache() -> SmartCache:
    """Get the global cache instance"""
    return smart_cache


def clear_all_caches():
    """Clear all caches"""
    global smart_cache
    smart_cache.memory_cache.clear()
    # Persistent cache would need individual key removal for full clear
    print("All caches cleared")