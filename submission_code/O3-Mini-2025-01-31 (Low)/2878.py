class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # diff will hold our difference array for the impact of subtract operations.
        diff = [0] * (n + 1)
        add = 0  # running sum of applied diff effects
        for i in range(n):
            add += diff[i]
            # effective value at index i after subtract operations
            effective = nums[i] + add
            if effective < 0:
                # if effective is negative, some previous excessive subtracts
                return False
            if effective > 0:
                # if i is too close to the end and we can't apply a full subarray operation, return False
                if i + k > n:
                    return False
                # apply subtract operations effective times
                add -= effective
                diff[i+k] += effective
        return True