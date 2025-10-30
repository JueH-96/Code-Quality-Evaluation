from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute the indices where x occurs in nums
        indices = []
        for i, num in enumerate(nums):
            if num == x:
                indices.append(i)
        
        # Process queries: for each query, find the (query[i])th occurrence (1-indexed)
        result = []
        for occ in queries:
            # If there are not enough occurrences, append -1
            if occ > len(indices):
                result.append(-1)
            else:
                result.append(indices[occ - 1])
        return result

# Example usage:
if __name__ == "__main__":
    solution = Solution()
    # Example 1
    nums = [1, 3, 1, 7]
    queries = [1, 3, 2, 4]
    x = 1
    print(solution.occurrencesOfElement(nums, queries, x))  # Output: [0, -1, 2, -1]

    # Example 2
    nums = [1, 2, 3]
    queries = [10]
    x = 5
    print(solution.occurrencesOfElement(nums, queries, x))  # Output: [-1]