class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in s
        s_indices = {}
        for i, char in enumerate(s):
            s_indices[char] = i
        
        # Calculate the permutation difference
        total_diff = 0
        for i, char in enumerate(t):
            # The index of char in s is s_indices[char]
            # The index of char in t is i
            total_diff += abs(s_indices[char] - i)
        
        return total_diff