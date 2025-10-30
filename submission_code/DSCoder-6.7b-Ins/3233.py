class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        # Initialize the count of partitions
        partitions = 0
        # Initialize the maximum length of prefix
        max_length = 0
        # Initialize the frequency of characters in the prefix
        freq = [0]*26
        
        # Iterate over the string
        for ch in s:
            # Increment the frequency of the character
            freq[ord(ch)-ord('a')] += 1
            # If the frequency of the character is less than or equal to k,
            # update the maximum length of prefix
            if freq[ord(ch)-ord('a')] <= k:
                max_length += 1
        
        # Iterate over the string again
        for i in range(len(s)):
            # If the character at index i is in the prefix,
            # increment the count of partitions and break the loop
            if freq[ord(s[i])-ord('a')] <= k and max_length > 0:
                partitions += 1
                max_length -= 1
        
        # Return the count of partitions
        return partitions