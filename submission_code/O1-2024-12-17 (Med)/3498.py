class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        half = n // 2

        # freq0[d] = how many of the "pairs" (nums[i], nums[n-i-1]) 
        # have absolute difference d
        freq0 = [0] * (k + 1)
        # MCount[m] = how many pairs have M = m, where
        # M = max(p, k-p, q, k-q) for the pair (p, q)
        MCount = [0] * (k + 1)

        # Build pair info
        for i in range(half):
            p, q = nums[i], nums[n - 1 - i]
            d = abs(p - q)
            freq0[d] += 1
            # M = max among (p, k-p, q, k-q)
            M = max(p, k - p, q, k - q)
            MCount[M] += 1

        # prefixM[x] will be the number of pairs whose M value is < x
        prefixM = [0] * (k + 1)
        for x in range(1, k + 1):
            prefixM[x] = prefixM[x - 1] + MCount[x - 1]

        # We use the formula:
        # cost(X) = number_of_pairs + (number_of_pairs whose M < X) - (number_of_pairs whose difference == X)
        #         = half + prefixM[X] - freq0[X]
        # Then we pick the X that minimizes this cost.
        res = float('inf')
        for x in range(k + 1):
            cost_x = half + prefixM[x] - freq0[x]
            if cost_x < res:
                res = cost_x

        return res