from typing import List

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Dictionary to store the count of k-sized subarrays each number appears in
        counts = {}

        # Iterate through all subarrays of size k
        # There are n - k + 1 such subarrays, starting at indices 0 to n - k
        for i in range(n - k + 1):
            # Get the current subarray
            subarray = nums[i : i + k]
            
            # Find unique numbers within the current subarray
            # We only need to count a number once per subarray it appears in
            unique_in_subarray = set(subarray)
            
            # Increment the count for each unique number found in this subarray
            for num in unique_in_subarray:
                counts[num] = counts.get(num, 0) + 1

        # Find the largest integer that appeared in exactly one subarray
        max_almost_missing = -1

        # Iterate through the collected counts
        # The keys of the counts dictionary are all numbers that appeared in at least one k-subarray
        for num, count in counts.items():
            # Check if the number appeared in exactly one subarray of size k
            if count == 1:
                # If it did, and it's larger than our current maximum, update the maximum
                max_almost_missing = max(max_almost_missing, num)

        return max_almost_missing