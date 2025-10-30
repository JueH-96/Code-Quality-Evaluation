class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Check each window of 3 consecutive tiles
        for i in range(n):
            # Get indices for 3 consecutive tiles considering circular array
            prev = (i - 1) % n
            curr = i
            next = (i + 1) % n
            
            # Check if middle tile is different from both neighbors
            if colors[curr] != colors[prev] and colors[curr] != colors[next]:
                count += 1
                
        return count