class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Gather all indices where nums[i] == x
        positions = []
        for i, val in enumerate(nums):
            if val == x:
                positions.append(i)

        # For each query, find the query-th occurrence in positions
        result = []
        for q in queries:
            # Check if the q-th occurrence exists (index q-1)
            if q <= len(positions):
                result.append(positions[q - 1])
            else:
                result.append(-1)

        return result