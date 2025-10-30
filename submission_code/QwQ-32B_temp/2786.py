class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n <= 1:
            return n
        
        # Compute the duplicates array
        duplicates = []
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                duplicates.append(1)
            else:
                duplicates.append(0)
        
        max_len = 1
        left = 0
        current_sum = 0
        
        for right in range(1, n):
            # Add the current duplicate (between right-1 and right)
            current_dup = duplicates[right - 1]
            current_sum += current_dup
            
            # Adjust left to maintain at most 1 duplicate
            while current_sum > 1:
                current_sum -= duplicates[left]
                left += 1
            
            # Update the maximum length
            current_length = right - left + 1
            if current_length > max_len:
                max_len = current_length
        
        return max_len