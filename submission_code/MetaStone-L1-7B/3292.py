class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        change_map = {}
        for s in range(len(changeIndices)):
            i = changeIndices[s]
            if i not in change_map:
                change_map[i] = []
            change_map[i].append(s + 1)  # Convert to 1-based second
        
        max_s_i = 0
        for i in range(1, len(nums) + 1):
            if i not in change_map:
                return -1
            required = nums[i-1]
            candidates = [s for s in change_map[i] if s >= required]
            if not candidates:
                return -1
            s_i = min(candidates)
            if s_i > max_s_i:
                max_s_i = s_i
        
        return max_s_i