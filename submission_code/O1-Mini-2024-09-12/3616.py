class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def simulate(nums_copy, start, direction):
            n = len(nums_copy)
            curr = start
            dir = direction  # -1 for left, 1 for right

            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    curr += dir
                elif nums_copy[curr] > 0:
                    nums_copy[curr] -= 1
                    dir *= -1  # Reverse direction
                    curr += dir
            return all(x == 0 for x in nums_copy)
        
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                for direction in [-1, 1]:  # -1: left, 1: right
                    nums_copy = nums.copy()
                    if simulate(nums_copy, i, direction):
                        count += 1
        return count