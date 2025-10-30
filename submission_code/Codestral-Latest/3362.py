class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from sortedcontainers import SortedList

        n = len(nums)
        distinct_counts = []

        # Use a sorted list to maintain the distinct elements in the current subarray
        current_subarray = SortedList()

        # Dictionary to keep track of the count of each element in the current subarray
        element_count = {}

        # Iterate over all possible subarrays
        for i in range(n):
            for j in range(i, n):
                if j > i:
                    # Remove the element that is no longer in the subarray
                    if element_count[nums[j - 1]] == 1:
                        current_subarray.remove(nums[j - 1])
                    element_count[nums[j - 1]] -= 1

                # Add the new element to the subarray
                if nums[j] in element_count:
                    element_count[nums[j]] += 1
                else:
                    element_count[nums[j]] = 1
                    current_subarray.add(nums[j])

                # Add the number of distinct elements in the current subarray to the list
                distinct_counts.append(len(current_subarray))

        # Sort the distinct_counts list
        distinct_counts.sort()

        # Find the median of the distinct_counts list
        median_index = len(distinct_counts) // 2
        if len(distinct_counts) % 2 == 0:
            median = (distinct_counts[median_index - 1] + distinct_counts[median_index]) // 2
        else:
            median = distinct_counts[median_index]

        return median