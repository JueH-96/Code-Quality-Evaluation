class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import Counter

        # Get the number of distinct elements in the entire array
        distinct_count = len(set(nums))

        # Initialize the result count
        complete_subarrays = 0

        # Iterate over all possible subarrays
        for start in range(len(nums)):
            current_counter = Counter()
            for end in range(start, len(nums)):
                current_counter[nums[end]] += 1
                if len(current_counter) == distinct_count:
                    complete_subarrays += 1

        return complete_subarrays