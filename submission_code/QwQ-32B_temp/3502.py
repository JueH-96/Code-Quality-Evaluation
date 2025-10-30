class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        if k == 1:
            return total  # All substrings are valid since every character appears at least once
        
        counts = [0] * 26
        left = 0
        invalid = 0
        num_invalid = 0
        
        for right in range(n):
            c = s[right]
            idx = ord(c) - ord('a')
            
            # Check if this character's count was exactly k-1 before incrementing
            if counts[idx] == k - 1:
                num_invalid += 1
            counts[idx] += 1
            
            # Shrink the window until all counts are <k
            while num_invalid > 0:
                left_c = s[left]
                left_idx = ord(left_c) - ord('a')
                
                if counts[left_idx] == k:
                    num_invalid -= 1
                counts[left_idx] -= 1
                left += 1
            
            # Add the number of valid substrings ending at 'right'
            invalid += (right - left + 1)
        
        return total - invalid