from pathlib import Path
import sqlite3
import hashlib
import json
import time
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Tuple, Any, Union
from datetime import datetime
import numpy as np
import io

class ArtRepository:
    def __init__(self, db_path: str = "art_repository.db", storage_path: str = "art_storage"):
        """Initialize the art repository system
        
        Args:
            db_path: Path to SQLite database
            storage_path: Directory for storing artwork files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize database with foreign key support
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.setup_database()
        
    def setup_database(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artwork (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                creator_id TEXT NOT NULL,
                creator_name TEXT NOT NULL,
                parent_id TEXT,
                timestamp REAL NOT NULL,
                tags TEXT NOT NULL,
                parameters TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                description TEXT,
                license TEXT DEFAULT 'CC BY-SA 4.0',
                views INTEGER DEFAULT 0,
                featured BOOLEAN DEFAULT 0,
                FOREIGN KEY(parent_id) REFERENCES artwork(id) ON DELETE SET NULL
            )
        ''')

    def store_artwork(self, image, title: str, creator_id: str, creator_name: str, **kwargs):
        if isinstance(image, bytes):
            content_hash = hashlib.sha256(image).hexdigest()
        else:
            content_hash = hashlib.sha256(image.getvalue()).hexdigest()
            
        artwork_id = f"{int(time.time())}_{content_hash[:8]}"
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO artwork 
            (id, title, creator_id, creator_name, timestamp, tags, parameters, storage_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (artwork_id, title, creator_id, creator_name, time.time(), json.dumps(kwargs.get('tags', [])), 
              json.dumps(kwargs.get('parameters', {})), str(kwargs.get('storage_path', ''))))
        return artwork_id

    
    def get_artwork(self, artwork_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """Retrieve artwork and its metadata"""
        cursor = self.conn.cursor()
        
        # Update view count
        cursor.execute('UPDATE artwork SET views = views + 1 WHERE id = ?', (artwork_id,))
        
        cursor.execute('SELECT * FROM artwork WHERE id = ?', (artwork_id,))
        row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Artwork {artwork_id} not found")
            
        # Load image
        image = Image.open(row[7])  # storage_path column
        
        # Build metadata
        metadata = {
            'id': row[0],
            'title': row[1],
            'creator': row[2],
            'parent_id': row[3],
            'timestamp': row[4],
            'tags': json.loads(row[5]),
            'parameters': json.loads(row[6]),
            'description': row[8],
            'license': row[9],
            'views': row[10],
            'featured': bool(row[11])
        }
        
        self.conn.commit()
        return image, metadata


    def update_artwork(self,
                      artwork_id: str,
                      title: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      description: Optional[str] = None,
                      license: Optional[str] = None) -> None:
        """Update artwork metadata
        
        Args:
            artwork_id: Artwork identifier
            title: New title
            tags: New tags list
            description: New description
            license: New license
        """
        updates = []
        params = []
        
        if title is not None:
            updates.append('title = ?')
            params.append(title)
        if tags is not None:
            updates.append('tags = ?')
            params.append(json.dumps(tags))
        if description is not None:
            updates.append('description = ?')
            params.append(description)
        if license is not None:
            updates.append('license = ?')
            params.append(license)
            
        if updates:
            params.append(artwork_id)
            query = f'''
                UPDATE artwork 
                SET {', '.join(updates)}
                WHERE id = ?
            '''
            
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
    
    def delete_artwork(self, artwork_id: str) -> None:
        """Delete artwork and its file
        
        Args:
            artwork_id: Artwork identifier
        """
        cursor = self.conn.cursor()
        
        # Get storage path before deletion
        cursor.execute('SELECT storage_path FROM artwork WHERE id = ?', (artwork_id,))
        row = cursor.fetchone()
        
        if row:
            # Delete file
            storage_path = Path(row[0])
            if storage_path.exists():
                storage_path.unlink()
            
            # Delete from database (cascades to versions and interactions)
            cursor.execute('DELETE FROM artwork WHERE id = ?', (artwork_id,))
            self.conn.commit()

    def get_artwork_history(self, artwork_id: str) -> List[Dict[str, Any]]:
        """Get modification history of artwork
        
        Args:
            artwork_id: Artwork identifier
            
        Returns:
            list: Version history with modifiers and timestamps
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT version_num, modifier, timestamp, changes
            FROM versions
            WHERE artwork_id = ?
            ORDER BY version_num
        ''', (artwork_id,))
        
        return [
            {
                'version': row[0],
                'modifier': row[1],
                'timestamp': row[2],
                'changes': json.loads(row[3])
            }
            for row in cursor.fetchall()
        ]
        
    def search_artwork(self, query: str, limit: int = 50):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, creator_id, creator_name, tags FROM artwork 
            WHERE id LIKE ? 
               OR lower(title) LIKE lower(?)
               OR lower(creator_id) LIKE lower(?) 
               OR lower(creator_name) LIKE lower(?)
               OR lower(tags) LIKE lower(?)
            ORDER BY timestamp DESC LIMIT ?
        ''', [f'%{query}%'] * 5 + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'title': row[1], 
                'creator_id': row[2],
                'creator_name': row[3],
                'tags': json.loads(row[4])
            })
        return results
    
    def _add_version(self, artwork_id: str, modifier: str, derived_id: str) -> None:
        """Add version record for artwork modification"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO versions
            (artwork_id, version_num, modifier, timestamp, changes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            artwork_id,
            self._get_next_version(artwork_id),
            modifier,
            time.time(),
            json.dumps({'derived_id': derived_id})
        ))
    
    def _get_next_version(self, artwork_id: str) -> int:
        """Get next version number for artwork modifications"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(version_num) FROM versions WHERE artwork_id = ?
        ''', (artwork_id,))
        current = cursor.fetchone()[0]
        return (current or 0) + 1
    
    def add_interaction(self,
                       artwork_id: str,
                       user_id: str,
                       interaction_type: str,
                       data: Optional[Dict[str, Any]] = None) -> None:
        """Record user interaction with artwork
        
        Args:
            artwork_id: Artwork identifier
            user_id: User identifier
            interaction_type: Type of interaction (like, comment, share, etc)
            data: Additional interaction data
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO interactions
            (artwork_id, user_id, interaction_type, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            artwork_id,
            user_id,
            interaction_type,
            time.time(),
            json.dumps(data or {})
        ))
        self.conn.commit()

    def get_trending_artwork(self,
                           timeframe: int = 86400,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending artwork based on recent interactions
        
        Args:
            timeframe: Time window in seconds (default 24 hours)
            limit: Maximum results to return
            
        Returns:
            list: Trending artwork with metadata and interaction counts
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                a.*,
                COUNT(DISTINCT i.user_id) as unique_users,
                COUNT(i.id) as total_interactions
            FROM artwork a
            LEFT JOIN interactions i ON a.id = i.artwork_id
            WHERE i.timestamp > ?
            GROUP BY a.id
            ORDER BY unique_users DESC, total_interactions DESC
            LIMIT ?
        ''', (time.time() - timeframe, limit))
        
        return [
            {
                'id': row[0],
                'title': row[1],
                'creator': row[2],
                'timestamp': row[4],
                'tags': json.loads(row[5]),
                'views': row[10],
                'featured': bool(row[11]),
                'unique_users': row[12],
                'total_interactions': row[13]
            }
            for row in cursor.fetchall()
        ]

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's artwork and interactions
        
        Args:
            user_id: User identifier
            
        Returns:
            dict: User statistics
        """
        cursor = self.conn.cursor()
        
        # Get artwork stats
        cursor.execute('''
            SELECT 
                COUNT(*) as artwork_count,
                SUM(views) as total_views,
                COUNT(CASE WHEN featured = 1 THEN 1 END) as featured_count
            FROM artwork
            WHERE creator = ?
        ''', (user_id,))
        
        artwork_stats = cursor.fetchone()
        
        # Get interaction stats
        cursor.execute('''
            SELECT 
                interaction_type,
                COUNT(*) as count
            FROM interactions
            WHERE artwork_id IN (SELECT id FROM artwork WHERE creator = ?)
            GROUP BY interaction_type
        ''', (user_id,))