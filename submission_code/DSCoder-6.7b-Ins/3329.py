class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert integers to strings
        arr1 = [str(i) for i in arr1]
        arr2 = [str(i) for i in arr2]
        
        # Initialize the longest common prefix
        lcp = ""
        
        # Iterate over the characters in the first string
        for i in range(len(arr1[0])):
            # Check if the current character is the same for all strings in arr1 and arr2
            if all(s[i] == arr1[0][0] for s in arr1) and all(s[i] == arr2[0][0] for s in arr2):
                # If yes, add the character to the longest common prefix
                lcp += arr1[0][0]
            else:
                # If not, break the loop as we have found the longest common prefix
                break
        
        # Return the length of the longest common prefix
        return len(lcp)