class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in string t.
        # This allows for O(1) average time lookup of character indices in t.
        # The dictionary comprehension iterates through t, getting the index 'i' and character 'char'
        # and stores char as the key and i as the value.
        t_indices = {char: i for i, char in enumerate(t)}

        # Initialize the total permutation difference.
        total_difference = 0

        # Iterate through the string s using enumerate to get both index and character.
        for i, char in enumerate(s):
            # 'i' is the index of the current character 'char' in string s.
            
            # Get the index of the same character 'char' in string t
            # by looking it up in the pre-computed dictionary.
            # This lookup is average O(1) time.
            j = t_indices[char]
            
            # Calculate the absolute difference between the index in s (i)
            # and the index in t (j) for the current character.
            # Add this absolute difference to the total difference.
            total_difference += abs(i - j)

        # Return the final calculated total permutation difference.
        return total_difference