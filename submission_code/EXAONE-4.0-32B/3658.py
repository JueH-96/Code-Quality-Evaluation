class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        fixed = [i for i in range(n) if nums[i] != -1]
        if not fixed:
            return 0
        candidates = []
        
        for i in range(1, len(fixed)):
            if fixed[i] == fixed[i-1] + 1:
                candidates.append(abs(nums[fixed[i]] - nums[fixed[i-1]]))
                
        for i in range(1, len(fixed)):
            if fixed[i] - fixed[i-1] > 1:
                k = fixed[i] - fixed[i-1] - 1
                diff = abs(nums[fixed[i]] - nums[fixed[i-1]])
                cand1 = (diff + k) // (k + 1)
                cand2 = (diff + 2) // 3
                candidates.append(max(cand1, cand2))
                
        return max(candidates) if candidates else 0