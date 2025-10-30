class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                for delta in [-1, 1]:
                    arr = nums.copy()
                    curr = i
                    direction = delta
                    while 0 <= curr < n:
                        if arr[curr] == 0:
                            curr += direction
                        else:
                            arr[curr] -= 1
                            direction *= -1
                            curr += direction
                    if all(x == 0 for x in arr):
                        count += 1
        return count