class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Loop over all possible starting points in the circular list
        for i in range(n):
            # For a circle, use modulo indexing
            left = i
            mid = (i + 1) % n
            right = (i + 2) % n
            
            # Check if the triplet forms an alternating group: x, y, x with x != y
            if colors[left] == colors[right] and colors[left] != colors[mid]:
                count += 1
        
        return count