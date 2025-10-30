class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # Calculate the number of segments
        m = len(word) // k
        
        # Create a dictionary to count the frequency of each segment
        freq = {}
        for i in range(m):
            segment = word[i * k : (i + 1) * k]
            freq[segment] = freq.get(segment, 0) + 1
        
        # The best candidate is the one with the maximum frequency.
        max_freq = max(freq.values())
        
        # To turn every segment into that candidate, we need to change the others.
        # Each operation can change one segment by copying from the candidate.
        return m - max_freq