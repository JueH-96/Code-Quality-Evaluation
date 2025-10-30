class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Check all triplets in the circle
        for i in range(n):
            # Get the indices of the triplet considering the circular nature
            first = colors[i]
            second = colors[(i + 1) % n]
            third = colors[(i + 2) % n]
            
            # Check if the triplet forms an alternating group
            if first != second and second != third and first == third:
                count += 1
        
        return count