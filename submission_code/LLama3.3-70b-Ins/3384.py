class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        count = {}
        
        # Count the frequency of each substring of length k
        for i in range(0, n, k):
            substring = word[i:i+k]
            if substring in count:
                count[substring] += 1
            else:
                count[substring] = 1
        
        # The minimum number of operations is the total number of substrings minus the frequency of the most frequent substring
        max_count = max(count.values())
        return len(count) - 1 if max_count == 1 else n // k - max_count