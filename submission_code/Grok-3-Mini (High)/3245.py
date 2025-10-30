class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def find_all(sub):
            indices = []
            start = 0
            while True:
                start = s.find(sub, start)
                if start == -1:
                    break
                indices.append(start)
                start += 1  # Allow overlapping matches
            return indices
        
        def has_in_range(nums, low, high):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] >= low and nums[mid] <= high:
                    return True
                elif nums[mid] < low:
                    left = mid + 1
                else:  # nums[mid] > high
                    right = mid - 1
            return False
        
        idx_a = find_all(a)
        idx_b = find_all(b)
        
        result = []
        for i in idx_a:
            if has_in_range(idx_b, i - k, i + k):
                result.append(i)
        
        return result