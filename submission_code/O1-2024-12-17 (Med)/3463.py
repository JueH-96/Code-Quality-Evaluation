class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        for i in range(n):
            # Next indices in the circle
            j = (i + 1) % n
            k = (i + 2) % n
            
            # Check the triplet (colors[i], colors[j], colors[k])
            # We need the pattern 0,1,0 or 1,0,1
            if (colors[i] == 0 and colors[j] == 1 and colors[k] == 0) or \
               (colors[i] == 1 and colors[j] == 0 and colors[k] == 1):
                count += 1
        
        return count