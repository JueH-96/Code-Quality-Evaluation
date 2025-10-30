from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # We are allowed to delete as many elements as we want.
        # Then, we choose a contiguous subarray of the resulting array that has distinct elements.
        # Notice that since we can delete arbitrarily,
        # we are completely free to remove duplicates and any elements that would hurt our total sum.
        # Hence, we can “construct” an array composed of one copy of each value from nums,
        # in the order they appear, with the freedom to drop any element.
        # Since the sum is what we care about (and order does not matter for the sum),
        # we should keep each positive number (and even zero if it does not hurt) 
        # and drop negative numbers (because they decrease the total sum).
        #
        # However if there are no positive numbers in the list,
        # any negative additions would reduce the sum. In that scenario,
        # the best we can do is pick the largest element (or zero, if present).
        
        pos_sum = 0
        has_positive = False
        for num in set(nums):
            if num > 0:
                pos_sum += num
                has_positive = True
                
        if has_positive:
            return pos_sum
        
        # If there are no positive numbers, the best option is to choose the maximum one.
        return max(nums)


# Below are some test cases:
if __name__ == '__main__':
    sol = Solution()
    # Example 1: All numbers are distinct and positive so we simply take them all.
    print(sol.maxSum([1,2,3,4,5]))  # Expected Output: 15

    # Example 2: The only positive candidate is 1.
    print(sol.maxSum([1,1,0,1,1]))  # Expected Output: 1

    # Example 3: Best is to keep distinct positive numbers {1, 2} (0 does not help and negatives hurt).
    print(sol.maxSum([1,2,-1,-2,1,0,-1]))  # Expected Output: 3

    # Additional test: Only zeros and negatives.
    print(sol.maxSum([0,0,-1,-2]))  # Expected Output: 0 (choosing 0 alone is best)

    # Additional test: All negatives.
    print(sol.maxSum([-5,-2,-7,-3]))  # Expected Output: -2 (largest among negatives)