class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # The number of operations needed is the sum of the number of changes needed for each position in the substring of length k.
        # We will count the frequency of each character at each position in the substrings of length k.
        # Then, for each position, we will find the character with the maximum frequency.
        # The number of changes needed for that position is the total number of substrings minus the maximum frequency.
        
        n = len(word)
        operations = 0
        
        # Loop through each position in the substring of length k
        for i in range(k):
            # Count the frequency of each character at this position
            freq = {}
            for j in range(i, n, k):
                freq[word[j]] = freq.get(word[j], 0) + 1
            
            # Find the character with the maximum frequency
            max_freq = max(freq.values())
            
            # Add the number of changes needed for this position to the total operations
            operations += (n // k) - max_freq
        
        return operations