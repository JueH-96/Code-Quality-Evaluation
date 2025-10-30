class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            s = set()
            imbalance = 0
            min_val, max_val = float('inf'), float('-inf')
            for j in range(i, n):
                if nums[j] in s:
                    continue
                s.add(nums[j])
                if len(s) > 1 and min_val < nums[j] - 1:
                    imbalance += 1
                if len(s) > 1 and max_val + 1 < nums[j]:
                    imbalance += 1
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                res += imbalance
        return res