class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Step 1: Create a dictionary to store the index of each character in t.
        # This allows for O(1) average time lookups for character indices in t.
        # For example, if t = "bac", t_char_to_index will be {'b': 0, 'a': 1, 'c': 2}.
        t_char_to_index = {char_val: index for index, char_val in enumerate(t)}
        
        # Step 2: Initialize the sum of permutation differences.
        current_permutation_difference = 0
        
        # Step 3: Iterate through string s to calculate differences.
        # For each character in s, we get its index in s (index_s)
        # and find its index in t (index_t) using the map.
        for index_s, char_s_val in enumerate(s):
            # Retrieve the index of char_s_val in string t from our map.
            # We are guaranteed that char_s_val exists in t_char_to_index because
            # t is a permutation of s and s has unique characters.
            index_t = t_char_to_index[char_s_val]
            
            # Calculate the absolute difference between the indices.
            difference = abs(index_s - index_t)
            
            # Add this difference to our running sum.
            current_permutation_difference += difference
            
        # Step 4: Return the total accumulated permutation difference.
        return current_permutation_difference