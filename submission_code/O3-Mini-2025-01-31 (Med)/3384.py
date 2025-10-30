class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # The word is composed of m = len(word)/k segments, each of length k.
        # We can copy any segment onto another segment in one operation.
        # Our goal is to make all segments equal to some string s (k-periodic).
        # The optimal strategy is to pick the segment that already appears
        # most frequently, and then copy it onto all other segments.
        # Thus, the minimum operations is (total segments - frequency of most common segment).
        
        m = len(word) // k  # How many segments of length k.
        segments = [word[i * k:(i + 1) * k] for i in range(m)]
        from collections import Counter
        freq = Counter(segments)
        max_freq = max(freq.values())
        
        return m - max_freq

# For testing purposes
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    word1 = "leetcodeleet"
    k1 = 4
    print(sol.minimumOperationsToMakeKPeriodic(word1, k1))  # Expected output: 1
    
    # Example 2:
    word2 = "leetcoleet"
    k2 = 2
    print(sol.minimumOperationsToMakeKPeriodic(word2, k2))  # Expected output: 3