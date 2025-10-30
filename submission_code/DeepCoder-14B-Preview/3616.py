class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] != 0:
                continue
            # Try right direction
            temp = list(nums)
            if self.simulate(i, 1, temp):
                count += 1
            # Try left direction
            temp = list(nums)
            if self.simulate(i, -1, temp):
                count += 1
        return count

    def simulate(self, start, initial_direction, nums_copy):
        n = len(nums_copy)
        curr = start
        direction = initial_direction
        while True:
            if curr < 0 or curr >= n:
                break
            if nums_copy[curr] == 0:
                new_curr = curr + direction
            else:
                nums_copy[curr] -= 1
                new_direction = -direction
                new_curr = curr + new_direction
                direction = new_direction
            curr = new_curr
        return all(x == 0 for x in nums_copy)