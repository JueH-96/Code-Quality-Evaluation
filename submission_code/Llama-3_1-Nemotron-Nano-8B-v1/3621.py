from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        if any(x < k for x in nums):
            return -1
        
        # Check if all elements are already k
        if all(x == k for x in nums):
            return 0
        
        # Collect unique elements >= k and sort in descending order
        unique_elements = sorted(list(set(nums)), reverse=True)
        unique_elements = [x for x in unique_elements if x >= k]
        
        if not unique_elements:
            return -1
        
        steps = 0
        
        while True:
            current_max = unique_elements[0]
            if current_max == k:
                return steps
            
            # Determine the next h
            if len(unique_elements) >= 2:
                h = unique_elements[1]
            else:
                h = k
            
            steps += 1
            
            # Compute new unique elements after applying h
            new_elements = [x for x in unique_elements if x <= h]
            if h not in new_elements:
                new_elements.append(h)
            new_elements.sort(reverse=True)
            
            unique_elements = new_elements
            
            # Check if we've reached k
            if unique_elements[0] == k:
                return steps
            
            # In case something goes wrong (shouldn't happen due to initial checks)
            if unique_elements[0] < k:
                return -1