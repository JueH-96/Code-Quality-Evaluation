from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute all indices where nums[i] == x
        positions = []
        for i, val in enumerate(nums):
            if val == x:
                positions.append(i)
        
        # For each query, retrieve the (k-1)-th occurrence if it exists
        answer = []
        for k in queries:
            if 1 <= k <= len(positions):
                answer.append(positions[k-1])
            else:
                answer.append(-1)
        
        return answer

# Example usage:
# sol = Solution()
# print(sol.occurrencesOfElement([1,3,1,7], [1,3,2,4], 1))  # [0, -1, 2, -1]
# print(sol.occurrencesOfElement([1,2,3], [10], 5))         # [-1]