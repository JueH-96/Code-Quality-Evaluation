class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        x, y = 0, 0
        max_dist = 0
        # We'll collect the best opportunities to change a character
        # Each opportunity is the potential gain in Manhattan distance
        opportunities = []
        
        for i, c in enumerate(s):
            original_x, original_y = x, y
            # Compute the original movement
            if c == 'N':
                y += 1
            elif c == 'S':
                y -= 1
            elif c == 'E':
                x += 1
            elif c == 'W':
                x -= 1
            original_dist = abs(x) + abs(y)
            
            # Now compute the best possible movement for this step
            best_dist = 0
            # Check all four possible directions
            directions = ['N', 'S', 'E', 'W']
            for d in directions:
                tx, ty = original_x, original_y
                if d == 'N':
                    ty += 1
                elif d == 'S':
                    ty -= 1
                elif d == 'E':
                    tx += 1
                elif d == 'W':
                    tx -= 1
                current_dist = abs(tx) + abs(ty)
                if current_dist > best_dist:
                    best_dist = current_dist
            
            gain = best_dist - original_dist
            if gain > 0:
                opportunities.append(gain)
            
            current_dist = abs(x) + abs(y)
            if current_dist > max_dist:
                max_dist = current_dist
        
        # We need to select top k gains
        opportunities.sort(reverse=True)
        total_gain = sum(opportunities[:k])
        max_dist += total_gain
        
        return max_dist