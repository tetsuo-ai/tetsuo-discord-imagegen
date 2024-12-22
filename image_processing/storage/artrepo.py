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
        """Initialize database with version and interaction tracking tables"""
        # First establish the database connection
        self.conn = sqlite3.connect(db_path)
        
        # Create storage directory if it doesn't exist
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Now we can create a cursor and set up tables
        cursor = self.conn.cursor()
        
        # Main artwork table
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
        
        # Version history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artwork_id TEXT NOT NULL,
                version_num INTEGER NOT NULL,
                modifier TEXT NOT NULL,
                timestamp REAL NOT NULL,
                changes TEXT NOT NULL,
                FOREIGN KEY(artwork_id) REFERENCES artwork(id) ON DELETE CASCADE
            )
        ''')
        
        # Interaction tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artwork_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                data TEXT,
                FOREIGN KEY(artwork_id) REFERENCES artwork(id) ON DELETE CASCADE
            )
        ''')
        
        self.conn.commit()

    def store_artwork(self, image: Union[bytes, Image.Image, BytesIO, str], 
                     title: str, creator_id: str, creator_name: str, **kwargs) -> str:
        """Store artwork with consistent handling of different input types
        
        Args:
            image: The image to store - can be:
                - bytes: Raw image data
                - PIL.Image: PIL Image object
                - BytesIO: BytesIO containing image data
                - str: Path to image file
            title: Artwork title
            creator_id: ID of creator
            creator_name: Name of creator
            **kwargs: Additional metadata including:
                - parent_id: ID of parent artwork (for remixes)
                - tags: List of tags
                - parameters: Dict of processing parameters
                - description: Text description
                
        Returns:
            str: Generated artwork ID
        """
        # Convert input to bytes consistently
        if isinstance(image, bytes):
            image_bytes = image
        elif isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        elif isinstance(image, BytesIO):
            image_bytes = image.getvalue()
        elif isinstance(image, str):
            with open(image, 'rb') as f:
                image_bytes = f.read()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Generate hash and ID
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        artwork_id = f"{int(time.time())}_{content_hash[:8]}"
        
        # Create storage filename and path
        storage_filename = f"{artwork_id}.png"
        storage_path = self.storage_path / storage_filename
        
        # Save image file
        with open(storage_path, 'wb') as f:
            f.write(image_bytes)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO artwork 
            (id, title, creator_id, creator_name, parent_id, timestamp, tags, parameters, storage_path, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            artwork_id,
            title,
            creator_id,
            creator_name,
            kwargs.get('parent_id'),
            time.time(),
            json.dumps(kwargs.get('tags', [])),
            json.dumps(kwargs.get('parameters', {})),
            str(storage_path),
            kwargs.get('description', '')
        ))
        
        self.conn.commit()
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
            
        # Load image from storage path
        stored_path = Path(row[8])  # storage_path column
        if not stored_path.exists():
            raise FileNotFoundError(f"Artwork file not found at {stored_path}")
            
        image = Image.open(stored_path)
        
        # Build metadata
        metadata = {
            'id': row[0],
            'title': row[1],
            'creator_id': row[2],
            'creator_name': row[3],
            'parent_id': row[4],
            'timestamp': row[5],
            'tags': json.loads(row[6]),
            'parameters': json.loads(row[7]),
            'description': row[9],
            'license': row[10],
            'views': row[11],
            'featured': bool(row[12])
        }
        
        self.conn.commit()
        return image, metadata

    def update_artwork(self, artwork_id: str, title: Optional[str] = None, 
                      tags: Optional[List[str]] = None, description: Optional[str] = None, 
                      license: Optional[str] = None) -> None:
        """Update artwork metadata"""
        ALLOWED_COLUMNS = {
            'title': str,
            'tags': list,
            'description': str,
            'license': str
        }
        
        updates = []
        params = []
        
        update_map = {
            'title': title,
            'tags': tags,
            'description': description,
            'license': license
        }
        
        for column, value in update_map.items():
            if value is not None:
                if column not in ALLOWED_COLUMNS:
                    raise ValueError(f"Invalid column name: {column}")
                
                updates.append(f'{column} = ?')
                params.append(json.dumps(value) if isinstance(value, list) else value)
        
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
        """Delete artwork and its file"""
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

    def add_version(self, artwork_id: str, modifier: str, changes: Dict[str, Any]) -> None:
        """Add a new version record for artwork modifications"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT MAX(version_num) FROM versions WHERE artwork_id = ?
        ''', (artwork_id,))
        current = cursor.fetchone()[0]
        next_version = (current or 0) + 1
        
        cursor.execute('''
            INSERT INTO versions 
            (artwork_id, version_num, modifier, timestamp, changes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            artwork_id,
            next_version,
            modifier,
            time.time(),
            json.dumps(changes)
        ))
        
        self.conn.commit()

    def add_interaction(self, artwork_id: str, user_id: str, 
                       interaction_type: str, data: Optional[Dict] = None) -> None:
        """Record a user interaction with artwork"""
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

    def get_artwork_history(self, artwork_id: str) -> List[Dict[str, Any]]:
        """Get modification history of artwork"""
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

    def search_artwork(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search artwork across all metadata fields"""
        cursor = self.conn.cursor()
        
        search_query = '''
            SELECT 
                id,
                title,
                creator_id,
                creator_name,
                timestamp,
                tags,
                description,
                views,
                parameters
            FROM artwork 
            WHERE id LIKE ? 
               OR lower(title) LIKE lower(?)
               OR lower(creator_id) LIKE lower(?)
               OR lower(creator_name) LIKE lower(?)
               OR lower(tags) LIKE lower(?)
               OR lower(description) LIKE lower(?)
               OR lower(parameters) LIKE lower(?)
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        search_term = f"%{query}%"
        params = [search_term] * 7 + [limit]
        
        cursor.execute(search_query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'title': row[1],
                'creator_id': row[2],
                'creator_name': row[3],
                'timestamp': row[4],
                'tags': json.loads(row[5]),
                'description': row[6],
                'views': row[7],
                'parameters': json.loads(row[8]) if row[8] else {}
            })
        
        return results

    def get_trending_artwork(self, timeframe: int = 86400, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending artwork based on recent interactions"""
        cursor = self.conn.cursor()
        
        query = '''
            WITH interaction_counts AS (
                SELECT 
                    artwork_id,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(CASE WHEN interaction_type = 'like' THEN 1 END) as likes,
                    COUNT(CASE WHEN interaction_type = 'remix' THEN 1 END) as remixes,
                    COUNT(CASE WHEN interaction_type = 'view' THEN 1 END) as views
                FROM interactions
                WHERE timestamp > ?
                GROUP BY artwork_id
            )
            SELECT 
                a.*,
                ic.unique_users,
                ic.likes,
                ic.remixes,
                ic.views,
                (ic.likes * 2 + ic.remixes * 3 + ic.views + ic.unique_users * 1.5) as trend_score
            FROM artwork a
            JOIN interaction_counts ic ON a.id = ic.artwork_id
            ORDER BY trend_score DESC
            LIMIT ?
        '''
        
        cursor.execute(query, (time.time() - timeframe, limit))
        
        results = []
        for row in cursor.fetchall():
            artwork_data = {
                'id': row[0],
                'title': row[1],
                'creator_id': row[2],
                'creator_name': row[3],
                'timestamp': row[5],
                'tags': json.loads(row[6]),
                'parameters': json.loads(row[7]),
                'views': row[11],
                'featured': bool(row[12]),
                'stats': {
                    'unique_users': row[-5],
                    'likes': row[-4],
                    'remixes': row[-3],
                    'recent_views': row[-2],
                    'trend_score': row[-1]
                }
            }
            results.append(artwork_data)
        
        return results

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's artwork and interactions"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as artwork_count,
                SUM(views) as total_views,
                COUNT(CASE WHEN featured = 1 THEN 1 END) as featured_count
            FROM artwork
            WHERE creator_id = ?
        ''', (user_id,))
        
        artwork_stats = cursor.fetchone()
        
        return {
            'artwork_count': artwork_stats[0] or 0,
            'total_views': artwork_stats[1] or 0,
            'featured_count': artwork_stats[2] or 0
        }
        
        
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    if hasattr(self, 'conn'):
        self.conn.close()