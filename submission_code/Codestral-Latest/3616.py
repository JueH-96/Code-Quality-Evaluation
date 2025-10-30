class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def simulate(start, direction):
            n = len(nums)
            curr = start
            nums_copy = nums[:]
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    if direction == 'right':
                        curr += 1
                    else:
                        curr -= 1
                elif nums_copy[curr] > 0:
                    nums_copy[curr] -= 1
                    direction = 'right' if direction == 'left' else 'left'
                    if direction == 'right':
                        curr += 1
                    else:
                        curr -= 1
            return all(x == 0 for x in nums_copy)

        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                if simulate(i, 'left'):
                    count += 1
                if simulate(i, 'right'):
                    count += 1
        return count