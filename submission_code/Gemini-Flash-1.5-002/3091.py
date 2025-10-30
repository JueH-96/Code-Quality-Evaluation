class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = 10**9 + 7
        count = 0
        n = len(nums)
        max_sum = sum(nums)
        dp = {}

        def backtrack(index, current_sum):
            if index == n:
                if l <= current_sum <= r:
                    return 1
                else:
                    return 0
            
            if (index, current_sum) in dp:
                return dp[(index, current_sum)]

            # Exclude the current element
            exclude = backtrack(index + 1, current_sum)

            # Include the current element (multiple times if possible)
            include = 0
            num_occurrences = nums.count(nums[index])
            for i in range(1, num_occurrences + 1):
                if current_sum + i * nums[index] <= max_sum:
                    include = (include + backtrack(index + 1, current_sum + i * nums[index])) % mod
                else:
                    break

            dp[(index, current_sum)] = (exclude + include) % mod
            return dp[(index, current_sum)]

        result = backtrack(0, 0)
        return result