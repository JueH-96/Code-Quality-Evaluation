class Solution:
    def minDifference(self, nums: List[int]) -> int:
        known = []
        n = len(nums)
        for i in range(n):
            if nums[i] != -1:
                known.append((i, nums[i]))
        
        if not known:
            return 0
        
        initial_max = 0
        for i in range(1, n):
            if nums[i-1] != -1 and nums[i] != -1:
                initial_max = max(initial_max, abs(nums[i] - nums[i-1]))
        
        max_required = 0
        for i in range(1, len(known)):
            a_idx, a_val = known[i-1]
            b_idx, b_val = known[i]
            diff = abs(b_val - a_val)
            distance = b_idx - a_idx
            if distance == 0:
                continue  # Consecutive elements, no gap
            required = diff // distance
            if diff % distance != 0:
                required += 1
            max_required = max(max_required, required)
        
        return max(initial_max, max_required)