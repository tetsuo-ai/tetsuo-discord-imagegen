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
        
        # Main artwork table (existing)
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

    def add_version(self, artwork_id: str, modifier: str, changes: Dict[str, Any]) -> None:
        """Add a new version record for artwork modifications
        
        Args:
            artwork_id: ID of modified artwork
            modifier: User ID who made the modification
            changes: Dictionary of changes made
        """
        cursor = self.conn.cursor()
        
        # Get next version number
        cursor.execute('''
            SELECT MAX(version_num) FROM versions WHERE artwork_id = ?
        ''', (artwork_id,))
        current = cursor.fetchone()[0]
        next_version = (current or 0) + 1
        
        # Insert version record
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
        """Record a user interaction with artwork
        
        Args:
            artwork_id: Artwork identifier
            user_id: User identifier
            interaction_type: Type of interaction (view, like, remix, etc)
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

    def get_trending_artwork(self, timeframe: int = 86400, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending artwork based on recent interactions
        
        Args:
            timeframe: Time window in seconds (default 24 hours)
            limit: Maximum results to return
            
        Returns:
            List of trending artwork with stats
        """
        cursor = self.conn.cursor()
        
        # Complex query to calculate trending score
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


    def update_artwork(self, artwork_id: str, title: Optional[str] = None, 
                      tags: Optional[List[str]] = None, description: Optional[str] = None, 
                      license: Optional[str] = None) -> None:
        # Define allowed columns and their types
        ALLOWED_COLUMNS = {
            'title': str,
            'tags': list,
            'description': str,
            'license': str
        }
        
        updates = []
        params = []
        
        # Use a dictionary to map parameters to their column names
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
        

def search_artwork(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search artwork across all metadata fields
    
    Args:
        query: Search term
        limit: Maximum number of results
        
    Returns:
        List of artwork metadata dictionaries
    """
    cursor = self.conn.cursor()
    
    # Create query to search across all relevant fields
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


    @bot.command(name='search')
    async def search_artwork(ctx, *, query: str = ""):
        """Search for artwork by any metadata"""
        try:
            if not query:
                await ctx.send("Please provide a search term!\n"
                             "Usage: !search <term>\n"
                             "Searches across: titles, tags, descriptions, creators, and effects")
                return

            results = art_repo.search_artwork(query)
            
            if not results:
                await ctx.send("No artwork found matching those terms.")
                return
                
            # Create paginated embed for results
            embed = discord.Embed(
                title=f"Search Results for '{query}'",
                description=f"Found {len(results)} matches",
                color=discord.Color.blue()
            )
            
            for art in results[:10]:  # Show first 10 results in first page
                # Format timestamp
                timestamp = datetime.fromtimestamp(art['timestamp'])
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                # Format effects/parameters if present
                effects = []
                for effect, value in art['parameters'].items():
                    if isinstance(value, tuple):
                        effects.append(f"{effect}: {value[0]}-{value[1]}")
                    else:
                        effects.append(f"{effect}: {value}")
                
                field_value = (
                    f"Creator: {art['creator_name']}\n"
                    f"Created: {time_str}\n"
                    f"Tags: {', '.join(art['tags'])}\n"
                    f"Views: {art['views']}\n"
                )
                
                if effects:
                    field_value += f"Effects: {', '.join(effects)}\n"
                    
                if art['description']:
                    field_value += f"Description: {art['description'][:100]}..."
                    
                embed.add_field(
                    name=f"{art['title']} (ID: {art['id']})",
                    value=field_value,
                    inline=False
                )
            
            if len(results) > 10:
                embed.set_footer(text=f"Showing 10 of {len(results)} results. Please refine your search for more specific results.")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"Error searching artwork: {str(e)}")
    
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