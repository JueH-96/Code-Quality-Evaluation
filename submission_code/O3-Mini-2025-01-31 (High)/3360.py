import collections

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count the frequency of each character.
        freq = collections.Counter(word)
        total = len(word)
        max_freq = max(freq.values())
        # Worst-case: delete everything (empty string is trivially k-special).
        ans = total
        
        # For each possible value r, we treat r as the maximum frequency that a
        # kept character will have after deletions.
        #
        # For any letter that we decide to "keep", we must have its original frequency
        # f >= L, where L is chosen so that the difference between r and L is at most k.
        # The best possible choice to keep as many letters as we can is to set L = max(1, r - k).
        #
        # Then, for every letter with frequency f:
        # - If f < L, we cannot use it (because we couldn’t “raise” its count), so we must delete all f occurrences.
        # - If f >= L, we can keep it. We want to lose as few characters as possible so we keep as many as allowed:
        #   If f ≥ r, we reduce its count to r (cost = f - r). If L ≤ f < r, we can leave it as is (cost = 0).
        
        for r in range(1, max_freq + 1):
            L = r - k
            if L < 1:
                L = 1
            cost = 0
            for f in freq.values():
                if f < L:
                    # f is too small to meet the lower bound L, so we have to remove all occurrences.
                    cost += f
                else:
                    # f is at least L.
                    # If f is larger than r, we cap it at r (cost = f - r). Otherwise, we keep it with 0 cost.
                    if f >= r:
                        cost += f - r
                    # When L <= f < r, cost is 0 (keep all counts since f is already between L and r).
            ans = min(ans, cost)
        
        return ans


# The following is for simple testing.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # "aabcaba" -> frequencies: a:4, b:2, c:1.
    # For k=0, to have all kept letters equal frequency,
    # one optimal solution is to delete two of the a's and one of the c's.
    # Expected output: 3
    print(sol.minimumDeletions("aabcaba", 0))
    
    # Example 2:
    # "dabdcbdcdcd" -> frequencies: d:5, a:1, b:2, c:3.
    # For k=2, one optimal solution is to remove one occurrence of "a" and one of "d".
    # Expected output: 2
    print(sol.minimumDeletions("dabdcbdcdcd", 2))
    
    # Example 3:
    # "aaabaaa" -> frequencies: a:6, b:1.
    # For k=2, one optimal solution is to delete the single "b"; the remaining all a's have frequency 6.
    # Expected output: 1
    print(sol.minimumDeletions("aaabaaa", 2))
    
    # Additional test: a word already k-special.
    print(sol.minimumDeletions("aabbcc", 10))  # Expected output: 0