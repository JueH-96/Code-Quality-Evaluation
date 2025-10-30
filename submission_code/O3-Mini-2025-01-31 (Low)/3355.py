from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # convert the possible list into score list: +1 for success (1), -1 for fail (0)
        scores = [1 if p == 1 else -1 for p in possible]
        total = sum(scores)
        alice_score = 0
        # Alice must play at least 1 level and Bob must play at least 1 level,
        # so we iterate k from 1 to n-1.
        for k in range(1, n):
            alice_score += scores[k-1]
            bob_score = total - alice_score
            if alice_score > bob_score:
                return k
        return -1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumLevels([1,0,1,0]))  # Expected output: 1
    print(sol.minimumLevels([1,1,1,1,1]))  # Expected output: 3
    print(sol.minimumLevels([0,0]))  # Expected output: -1