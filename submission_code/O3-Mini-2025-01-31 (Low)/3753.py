class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequencies of each character
        freq = Counter(s)
        odd_list = []
        even_list = []
        
        # Separate frequencies into odd and even lists
        for count in freq.values():
            if count % 2 == 0:
                even_list.append(count)
            else:
                odd_list.append(count)
        
        # The maximum difference will be max(odd) - min(even)
        max_odd = max(odd_list)
        min_even = min(even_list)
        
        return max_odd - min_even

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test cases
    print(sol.maxDifference("aaaaabbc"))    # Expected output: 3
    print(sol.maxDifference("abcabcab"))    # Expected output: 1