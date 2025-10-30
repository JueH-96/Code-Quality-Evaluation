class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        for i in range(n):
            # Check the current triplet (i, (i+1) % n, (i+2) % n)
            if colors[i] != colors[(i + 1) % n] and colors[i] != colors[(i + 2) % n]:
                count += 1
        
        return count