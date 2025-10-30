class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}

        def calculate_strength(subarrays):
            strength = 0
            for i, sub in enumerate(subarrays):
                strength += (-1)**(i) * sum(nums[sub[0]:sub[1]+1]) * (len(subarrays) - i)
            return strength

        def find_max_strength(index, remaining_k, current_subarrays):
            if remaining_k == 0:
                return calculate_strength(current_subarrays)
            if index == n:
                return -float('inf')
            if (index, remaining_k, tuple(tuple(sub) for sub in current_subarrays)) in dp:
                return dp[(index, remaining_k, tuple(tuple(sub) for sub in current_subarrays))]

            max_strength = -float('inf')
            
            # Option 1: Don't include current element in any subarray
            max_strength = max(max_strength, find_max_strength(index + 1, remaining_k, current_subarrays))

            # Option 2: Include current element in a new subarray
            if remaining_k > 0:
                for j in range(index, n):
                    new_subarrays = current_subarrays + [[index, j]]
                    max_strength = max(max_strength, find_max_strength(j + 1, remaining_k - 1, new_subarrays))

            dp[(index, remaining_k, tuple(tuple(sub) for sub in current_subarrays))] = max_strength
            return max_strength

        result = find_max_strength(0, k, [])
        return result