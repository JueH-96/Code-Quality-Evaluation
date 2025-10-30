from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        # Since the array contains n-2 special numbers, one sum, and one outlier,
        # we need to find the largest number that cannot be the sum of any two special numbers.
        
        # To find the largest potential outlier, we can consider the largest element in the array.
        # If it cannot be expressed as the sum of any two other elements, it is the outlier.
        
        # However, considering the constraints, we need an efficient way to handle this.
        # Given that the array can be up to 10^5 elements, we need a solution that is O(n) or O(n log n).
        
        # One approach is to sort the array and then check the largest element to see if it can be the sum of any two other elements.
        
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        
        # The largest element is the last one in the sorted list
        largest = nums_sorted[-1]
        
        # Now, we need to check if the largest can be the sum of any two elements in the array
        # We can use a two-pointer approach to find if any two elements sum to the largest
        
        left = 0
        right = n - 2  # since we are excluding the last element
        
        while left < right:
            current_sum = nums_sorted[left] + nums_sorted[right]
            if current_sum == largest:
                # If the sum is equal to the largest, then the largest is not the outlier
                # So we need to find the next largest element that cannot be the sum
                # We can try the next largest element
                # So we move the right pointer to the left
                right -= 1
            elif current_sum < largest:
                left += 1
            else:
                right -= 1
        
        # After the loop, if no two elements sum to the largest, then the largest is the outlier
        # Otherwise, we need to find the next largest element that cannot be the sum
        # So we can iterate from the end and find the first element that cannot be the sum of any two elements
        
        # To optimize, we can precompute all possible sums and then find the largest element not in the sum set
        # But given the constraints, it's better to use the two-pointer approach
        
        # Since the above loop only checks for the largest, we need to ensure that the largest is indeed the outlier
        # If not, we need to find the next largest
        
        # So, we can iterate from the end and find the first element that cannot be the sum of any two elements
        for i in range(n-1, -1, -1):
            candidate = nums_sorted[i]
            left = 0
            right = n - 1
            while left < right:
                if left == i:
                    left += 1
                    continue
                if right == i:
                    right -= 1
                    continue
                current_sum = nums_sorted[left] + nums_sorted[right]
                if current_sum == candidate:
                    break
                elif current_sum < candidate:
                    left += 1
                else:
                    right -= 1
            else:
                # If no two elements sum to candidate, then it is the outlier
                return candidate
        
        # If all elements can be expressed as the sum of two others, then there is no outlier
        # But the problem states that at least one potential outlier exists
        return -1