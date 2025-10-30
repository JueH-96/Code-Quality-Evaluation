class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        zeros = [i for i in range(n) if nums[i] == 0]
        
        for start_curr in zeros:
            for direction in [1, -1]:  # 1 for right, -1 for left
                arr = nums.copy()
                curr = start_curr
                delta = direction
                while 0 <= curr < n:
                    if arr[curr] == 0:
                        curr += delta
                    else:
                        arr[curr] -= 1
                        delta = -delta
                        curr += delta
                if all(x == 0 for x in arr):
                    count += 1
        
        return count