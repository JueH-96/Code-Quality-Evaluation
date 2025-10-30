class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        # Sliding window approach:
        n = len(s)
        count = {}
        left = 0
        max_len = 0
        
        # Iterate with right pointer
        for right in range(n):
            char = s[right]
            count[char] = count.get(char, 0) + 1
            
            # While any character in the window exceeds 2 occurrences
            while count[char] > 2:
                # Reduce count at the left side of the window
                left_char = s[left]
                count[left_char] -= 1
                left += 1
            
            # Update the maximum length found
            max_len = max(max_len, right - left + 1)
        
        return max_len

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumLengthSubstring("bcbbbcba"))  # Expected output: 4
    print(sol.maximumLengthSubstring("aaaa"))      # Expected output: 2