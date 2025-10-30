class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        primes_set = {
            1: set(),
            2: {2},
            3: {3},
            4: {2},
            5: {5},
            6: {2, 3},
            7: {7},
            8: {2},
            9: {3},
            10: {2, 5}
        }
        max_len = 0
        for i in range(n):
            current_primes = set()
            is_pairwise = True
            for j in range(i, n):
                num = nums[j]
                factors = primes_set[num]
                if j == i:
                    if num == 1:
                        max_len = max(max_len, 1)
                    current_primes = factors.copy()
                else:
                    if factors & current_primes:
                        if j == i + 1:
                            max_len = max(max_len, 2)
                            current_primes |= factors
                            is_pairwise = False
                        else:
                            break
                    else:
                        current_primes |= factors
                        if is_pairwise:
                            max_len = max(max_len, j - i + 1)
        return max_len