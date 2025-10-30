from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Initialize the prefix and suffix arrays
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)

        # Calculate the prefix array
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]

        # Calculate the suffix array
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + nums[i]

        # Initialize the count of monotonic pairs
        count = 0

        # Iterate over all possible values of arr1[0]
        for x in range(nums[0] + 1):
            arr1 = [x]
            arr2 = [nums[0] - x]
            valid = True

            # Build arr1 and arr2 based on the initial value x
            for i in range(1, n):
                y = min(nums[i], arr1[-1])
                arr1.append(y)
                arr2.append(nums[i] - y)
                if arr2[-1] > arr2[-2]:
                    valid = False
                    break

            # If the constructed arrays are valid, count the pairs
            if valid:
                count = (count + 1) % MOD

        return count