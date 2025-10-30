class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        
        def find_dominant(arr):
            if not arr:
                return None
            counts = {}
            for x in arr:
                counts[x] = counts.get(x, 0) + 1
            
            dominant = None
            for x, count in counts.items():
                if count * 2 > len(arr):
                    dominant = x
                    break
            return dominant

        dominant_all = find_dominant(nums)
        
        for i in range(n - 1):
            left_arr = nums[:i+1]
            right_arr = nums[i+1:]
            
            dominant_left = find_dominant(left_arr)
            dominant_right = find_dominant(right_arr)
            
            if dominant_left == dominant_all and dominant_right == dominant_all:
                return i
        
        return -1