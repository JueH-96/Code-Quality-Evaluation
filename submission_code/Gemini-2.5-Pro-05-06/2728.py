from typing import List

class Solution:
  def matrixSum(self, nums: List[List[int]]) -> int:
    # Step 1: Sort each row in ascending order.
    # This ensures that the largest element in any row is at the end of its list.
    for row in nums:
        row.sort()

    score = 0
    
    # Determine the number of operations to perform.
    # This will be equal to the length of the longest row in the matrix.
    # Constraints: nums.length >= 1, nums[i].length >= 1.
    # So nums is not empty and rows are not empty initially.
    # Therefore, num_operations will be at least 1.
    num_operations = 0
    if not nums: # Should not be strictly necessary given constraints but good for robustness
        return 0 
    # Calculate the maximum length among all rows.
    # Using a generator expression with max() is a concise way to do this.
    num_operations = max(len(r) for r in nums)

    # Perform the operations num_operations times.
    # In each operation:
    # 1. Remove the largest element from each row (if the row is not empty).
    # 2. Find the maximum among these removed elements.
    # 3. Add this maximum to the score.
    for _ in range(num_operations):
        max_element_this_step = 0 # Since nums[i][j] >= 0, 0 is a valid initial minimum.
        
        for row in nums:
            # If the row is not empty, it means there are elements to remove.
            if row: # Pythonic way to check if a list is not empty
                # .pop() removes and returns the last element from the list.
                # Since rows are sorted, this is the largest remaining element in this row.
                element_removed = row.pop()
                if element_removed > max_element_this_step:
                    max_element_this_step = element_removed
            # If a row is empty, we skip it for this step. It cannot contribute an element.
            
        score += max_element_this_step
            
    return score