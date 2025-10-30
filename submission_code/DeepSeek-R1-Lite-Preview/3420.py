from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Step 1: Collect indices where nums[i] == x
        x_indices = [i for i, num in enumerate(nums) if num == x]
        
        # Step 2: Answer each query
        answer = [x_indices[q - 1] if q <= len(x_indices) else -1 for q in queries]
        
        return answer