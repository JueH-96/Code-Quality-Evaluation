class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Check for alternating groups including the wrap-around case
        for i in range(n):
            # Use modulo to wrap around the circle
            prev = colors[i - 1]
            curr = colors[i]
            next = colors[(i + 1) % n]
            
            # Check if the current group is alternating
            if curr != prev and curr != next and prev == next:
                count += 1
        
        return count