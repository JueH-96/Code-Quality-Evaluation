class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute prefix max
        prefix_max = [nums[0]]
        for num in nums[1:]:
            prefix_max.append(max(prefix_max[-1], num))
        # Compute suffix max
        suffix_max = [0] * n
        suffix_max[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_max[i] = max(nums[i], suffix_max[i+1])
        # Initialize max value to 0
        max_val = 0
        # Iterate over j from 1 to n-2 inclusive
        for j in range(1, n-1):
            diff = prefix_max[j-1] - nums[j]
            if diff > 0:
                prod = diff * suffix_max[j+1]
                if prod > max_val:
                    max_val = prod
        return max_val