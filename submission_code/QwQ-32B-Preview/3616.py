class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def simulate(start, direction, nums):
            curr = start
            n = len(nums)
            while 0 <= curr < n:
                if nums[curr] == 0:
                    if direction == 'left':
                        curr -= 1
                    else:
                        curr += 1
                elif nums[curr] > 0:
                    nums[curr] -= 1
                    if direction == 'left':
                        direction = 'right'
                    else:
                        direction = 'left'
                    if direction == 'left':
                        curr -= 1
                    else:
                        curr += 1
            return all(x == 0 for x in nums)
        
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                # Try starting at i with direction 'left'
                if simulate(i, 'left', nums.copy()):
                    count += 1
                # Try starting at i with direction 'right'
                if simulate(i, 'right', nums.copy()):
                    count += 1
        return count