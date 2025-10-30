class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Create a dictionary to store the indices of each character in str1
        char_indices = {}
        for i, char in enumerate(str1):
            if char not in char_indices:
                char_indices[char] = []
            char_indices[char].append(i)
        
        # Iterate through str2 and try to find a matching character in str1
        prev_index = -1
        for char in str2:
            if char not in char_indices:
                return False
            
            # Find the index of the next character in str1 that is greater than the previous index
            for index in char_indices[char]:
                if index > prev_index:
                    prev_index = index
                    break
            else:
                return False
        
        return True