class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        
        def simulate(target_moves):
            # target_moves is a list of preferred moves in order of priority
            changes = 0
            x, y = 0, 0
            max_dist = 0
            
            for i, move in enumerate(s):
                # Decide what move to make
                actual_move = move
                
                # If we can still make changes, try to use a preferred move
                if changes < k:
                    for preferred in target_moves:
                        if preferred != move:
                            actual_move = preferred
                            changes += 1
                            break
                
                # Make the move
                if actual_move == 'N':
                    y += 1
                elif actual_move == 'S':
                    y -= 1
                elif actual_move == 'E':
                    x += 1
                elif actual_move == 'W':
                    x -= 1
                
                # Update max distance
                max_dist = max(max_dist, abs(x) + abs(y))
            
            return max_dist
        
        # Try different strategies
        strategies = [
            ['N'],  # Go north as much as possible
            ['S'],  # Go south as much as possible
            ['E'],  # Go east as much as possible
            ['W'],  # Go west as much as possible
            ['N', 'E'],  # Prefer north, then east
            ['N', 'W'],  # Prefer north, then west
            ['S', 'E'],  # Prefer south, then east
            ['S', 'W'],  # Prefer south, then west
            ['E', 'N'],  # Prefer east, then north
            ['E', 'S'],  # Prefer east, then south
            ['W', 'N'],  # Prefer west, then north
            ['W', 'S'],  # Prefer west, then south
        ]
        
        max_result = 0
        for strategy in strategies:
            max_result = max(max_result, simulate(strategy))
        
        return max_result