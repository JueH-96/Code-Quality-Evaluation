class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        max_distance = 0
        
        # Iterate through the string
        for i in range(n):
            # Count the number of changes needed to the left of the current position
            left_changes = 0
            for j in range(i-1, -1, -1):
                if s[j] != s[i]:
                    left_changes += 1
                    if left_changes > k:
                        break
            
            # Count the number of changes needed to the right of the current position
            right_changes = 0
            for j in range(i+1, n):
                if s[j] != s[i]:
                    right_changes += 1
                    if right_changes > k:
                        break
            
            # Calculate the Manhattan distance and update the maximum
            distance = left_changes + right_changes + 1
            max_distance = max(max_distance, distance)
        
        return max_distance