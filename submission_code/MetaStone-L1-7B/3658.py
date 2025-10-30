class Solution:
    def minDifference(self, nums: List[int]) -> int:
        known = []
        for num in nums:
            if num != -1:
                known.append(num)
        max_diff = 0
        for i in range(len(known) - 1):
            a = known[i]
            b = known[i + 1]
            diff = abs(a - b)
            d_i = (diff + 1) // 2
            if d_i > max_diff:
                max_diff = d_i
        return max_diff