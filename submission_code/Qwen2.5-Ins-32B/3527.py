from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        colors += colors  # Extend colors to handle circular nature
        count = [0] * (n + 1)
        
        # Precompute the count of alternating groups for each size
        for size in range(3, n + 1):
            for i in range(n):
                if (colors[i] != colors[i + 1] and 
                    colors[i + 1] != colors[i + 2] and 
                    colors[i + size - 1] != colors[i + size - 2]):
                    count[size] += 1
        
        result = []
        for query in queries:
            if query[0] == 1:
                result.append(count[query[1]])
            else:
                index, color = query[1], query[2]
                colors[index] = color
                colors[index + n] = color  # Update the extended part as well
                
                # Recompute the count of alternating groups for affected sizes
                for size in range(3, n + 1):
                    for i in range(max(0, index - size + 1), min(index + 2, n)):
                        if (colors[i] != colors[i + 1] and 
                            colors[i + 1] != colors[i + 2] and 
                            colors[i + size - 1] != colors[i + size - 2]):
                            count[size] += 1
                        else:
                            count[size] -= 1
        
        return result