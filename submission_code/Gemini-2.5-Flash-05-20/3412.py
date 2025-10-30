class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in string t.
        # This allows for O(1) average-time lookups for character positions in t.
        t_char_map = {}
        for i, char in enumerate(t):
            t_char_map[char] = i
        
        # Initialize the total permutation difference.
        total_diff = 0
        
        # Iterate through string s. For each character s[i]:
        # 1. Its index in s is 'i'.
        # 2. Its index in t is retrieved from the t_char_map.
        # 3. Calculate the absolute difference between these two indices and add it to total_diff.
        for i, char_s in enumerate(s):
            index_s = i
            index_t = t_char_map[char_s] # Get the index of char_s from string t
            
            total_diff += abs(index_s - index_t)
            
        # Return the accumulated total difference.
        return total_diff