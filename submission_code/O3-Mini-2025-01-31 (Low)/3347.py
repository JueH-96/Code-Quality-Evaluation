from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # initialize arr1 and arr2 with the first two elements
        arr1 = [nums[0]]
        arr2 = [nums[1]]
        
        # process remaining elements from index 2 (3rd element) to end
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
                
        # concatenate arr1 and arr2 and return the result
        return arr1 + arr2

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test Example 1
    print(sol.resultArray([2, 1, 3]))  # Expected output: [2, 3, 1]
    # Test Example 2
    print(sol.resultArray([5, 4, 3, 8]))  # Expected output: [5, 3, 4, 8]