class Solution:
    def maxDifference(self, s: str) -> int:
        # Count frequency of each character in the string
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Collect characters with odd and even frequencies separately
        odd_chars = []
        even_chars = []
        for char, count in freq.items():
            if count % 2 == 1:
                odd_chars.append(count)
            else:
                even_chars.append(count)
        
        # Since the constraints guarantee there is at least one odd and one even frequency,
        # we iterate over all combinations to find the maximum difference as odd - even.
        max_diff = -float("inf")
        for odd_f in odd_chars:
            for even_f in even_chars:
                diff = odd_f - even_f
                if diff > max_diff:
                    max_diff = diff
        
        return max_diff