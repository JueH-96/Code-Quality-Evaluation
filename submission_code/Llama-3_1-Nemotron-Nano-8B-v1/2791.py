from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        if n == 0:
            return []
        
        visited = set()
        current = 1
        visited.add(current)
        step = 1  # The multiplier for k, starting at 1 for the first step
        
        while True:
            # Calculate the next position
            next_pos = (current - 1 + step * k) % n + 1
            
            if next_pos in visited:
                break  # Game ends as next_pos is revisited
            
            visited.add(next_pos)
            current = next_pos
            step += 1
        
        # Determine the losers by subtracting visited from all friends
        all_friends = set(range(1, n + 1))
        losers = sorted(all_friends - visited)
        
        return losers