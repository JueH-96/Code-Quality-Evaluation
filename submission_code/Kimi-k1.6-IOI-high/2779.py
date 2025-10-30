class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        colors = [0] * n
        count = 0
        answer = []
        for idx, color in queries:
            old_color = colors[idx]
            if old_color == color:
                answer.append(count)
                continue
            
            # Subtract contributions from the old color
            for dx in (-1, 1):
                neighbor = idx + dx
                if 0 <= neighbor < n:
                    if old_color != 0 and colors[neighbor] == old_color:
                        count -= 1
            
            # Update the color
            colors[idx] = color
            
            # Add contributions from the new color
            for dx in (-1, 1):
                neighbor = idx + dx
                if 0 <= neighbor < n:
                    if colors[neighbor] == color:
                        count += 1
            
            answer.append(count)
        
        return answer