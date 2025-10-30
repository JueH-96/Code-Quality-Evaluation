class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for i in range(min(n, limit) + 1):
            for j in range(min(n - i, limit) + 1):
                k = n - i - j
                if 0 <= k <= limit:
                    # Since the order of children matters, we count all permutations
                    if i == j == k:
                        count += 1  # (1, 1, 1) is counted once
                    elif i == j or j == k or i == k:
                        count += 3  # (1, 2, 1) is counted 3 times
                    else:
                        count += 6  # (1, 2, 3) is counted 6 times
        return count