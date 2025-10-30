class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        segments = []
        current_seg = [nums[0]]
        current_pop = bin(nums[0]).count('1')
        
        for i in range(1, n):
            num = nums[i]
            pop_count = bin(num).count('1')
            if pop_count == current_pop:
                current_seg.append(num)
            else:
                segments.append(current_seg)
                current_seg = [num]
                current_pop = pop_count
        segments.append(current_seg)
        
        sorted_segments = [sorted(seg) for seg in segments]
        
        result = []
        for seg in sorted_segments:
            result.extend(seg)
            
        for i in range(1, len(result)):
            if result[i] < result[i-1]:
                return False
        return True