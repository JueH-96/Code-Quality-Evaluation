class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        diffs1 = []
        diffs2 = []
        for i in range(n):
            diff1 = nums[i] - (nums[i] + 1) // 2
            diffs1.append(diff1)
            if nums[i] >= k:
                diff2 = k
                diffs2.append(diff2)
            else:
                diffs2.append(0)
        
        diffs1.sort(reverse=True)
        diffs2.sort(reverse=True)

        total_sum = sum(nums)
        
        for i in range(min(op1, n)):
            total_sum -= diffs1[i]
            
        for i in range(min(op2, n)):
            total_sum -= diffs2[i]
            
        return total_sum