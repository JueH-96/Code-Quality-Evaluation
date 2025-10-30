class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # We will use a "difference array" approach to keep track of how many flips affect each position.
        diff = [0]*(n+1)  # Extra space to handle range updates safely.
        current_flips = 0  # Running sum of flips in effect at the current index.
        operations = 0

        for i in range(n):
            # Update current_flips by adding whatever was "scheduled" to start/end at this index
            current_flips += diff[i]

            # Effective value = original nums[i] flipped current_flips times
            effective_val = (nums[i] + current_flips) % 2
            if effective_val == 0:  # If it's 0, we need to flip i, i+1, i+2
                if i+2 >= n:  # Can't flip 3 consecutive if not enough space
                    return -1
                operations += 1
                # We flip from i to i+2, so record that we started a flip at i
                current_flips += 1
                # We also mark where the flip effect ends (just after i+2)
                diff[i+3] -= 1

        # After processing all indices, if the array is valid (all 1s), return operations
        # If any final elements were impossible to correct, they'd have triggered return -1 earlier.
        return operations