class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0

        def simulate(start_index, direction):
            temp_nums = nums[:]
            curr = start_index
            while 0 <= curr < n:
                if temp_nums[curr] == 0:
                    curr += direction
                else:
                    temp_nums[curr] -= 1
                    direction *= -1
                    curr += direction
            return all(x == 0 for x in temp_nums)

        for i in range(n):
            if nums[i] == 0:
                if simulate(i, 1):
                    ans += 1
                if simulate(i, -1):
                    ans += 1
        return ans