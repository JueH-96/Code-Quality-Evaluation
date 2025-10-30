from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total = 0
        for seq in combinations(nums, k):
            min_diff = float('inf')
            for i in range(len(seq)):
                for j in range(i+1, len(seq)):
                    current_diff = abs(seq[i] - seq[j])
                    if current_diff < min_diff:
                        min_diff = current_diff
            total += min_diff
        return total % MOD