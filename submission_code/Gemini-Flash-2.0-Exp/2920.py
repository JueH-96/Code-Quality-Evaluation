class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if len(set(nums)) == 1:
            return 0
        
        indices = {}
        for i in range(n):
            if nums[i] not in indices:
                indices[nums[i]] = []
            indices[nums[i]].append(i)
        
        max_dist = 0
        for num in indices:
            positions = indices[num]
            max_diff = 0
            for i in range(len(positions)):
                diff = (positions[(i + 1) % len(positions)] - positions[i] - 1) % n
                max_diff = max(max_diff, diff)
            max_dist = max(max_dist, (max_diff + 1) // 2)
        
        return max_dist