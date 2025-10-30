class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # First, collect all the indices where nums[i] == x
        positions = []
        for i, val in enumerate(nums):
            if val == x:
                positions.append(i)

        # For each query, check if the occurrence is valid
        # queries[i] is the 1-based occurrence count
        result = []
        for q in queries:
            # If q exceeds the number of occurrences, return -1
            if q > len(positions):
                result.append(-1)
            else:
                # Otherwise, return the (q-1)th element in positions
                result.append(positions[q - 1])

        return result