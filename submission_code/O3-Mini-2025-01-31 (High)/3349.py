class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        # Using a sliding window approach with a frequency dictionary.
        from collections import defaultdict
        
        left = 0
        max_len = 0
        freq = defaultdict(int)
        
        # Iterate with right pointer over the string
        for right, char in enumerate(s):
            freq[char] += 1
            # If any character exceeds two occurrences,
            # shrink the window from the left until it's valid.
            while freq[char] > 2:
                freq[s[left]] -= 1
                left += 1
            # Update the maximum length found so far.
            max_len = max(max_len, right - left + 1)
            
        return max_len


# You can use the following test cases to verify the solution:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    s1 = "bcbbbcba"
    print(sol.maximumLengthSubstring(s1))  # Expected output: 4

    # Example 2:
    s2 = "aaaa"
    print(sol.maximumLengthSubstring(s2))  # Expected output: 2