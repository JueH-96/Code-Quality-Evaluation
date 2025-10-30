class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        colors = [0] * n
        result = []
        adjacentCount = 0
        
        for index, color in queries:
            oldColor = colors[index]
            colors[index] = color
            
            # Check both directions for adjacent colors
            if index > 0 and colors[index] == colors[index - 1] and oldColor != colors[index]:
                if oldColor != colors[index - 1]:
                    adjacentCount += 1
                elif oldColor == 0 and colors[index - 1] == color:
                    adjacentCount += 1
                else:
                    adjacentCount -= 1
                    
            if index < n - 1 and colors[index] == colors[index + 1] and oldColor != colors[index]:
                if oldColor != colors[index + 1]:
                    adjacentCount += 1
                elif oldColor == 0 and colors[index + 1] == color:
                    adjacentCount += 1
                else:
                    adjacentCount -= 1

            result.append(adjacentCount)

        return result