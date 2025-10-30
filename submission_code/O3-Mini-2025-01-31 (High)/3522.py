from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        # Iterate for every subarray window of size k
        for i in range(n - k + 1):
            valid = True
            # Check if the subarray is sorted in ascending order with consecutive differences of 1
            for j in range(i, i + k - 1):
                if nums[j+1] - nums[j] != 1:
                    valid = False
                    break
            if valid:
                # The array is ascending with consecutive elements so the maximum is the last element
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        return results

# Example usage:
if __name__ == "__main__":
    solution = Solution()
    print(solution.resultsArray([1,2,3,4,3,2,5], 3))  # Expected output: [3, 4, -1, -1, -1]
    print(solution.resultsArray([2,2,2,2,2], 4))       # Expected output: [-1, -1]
    print(solution.resultsArray([3,2,3,2,3,2], 2))       # Expected output: [-1, 3, -1, 3, -1]