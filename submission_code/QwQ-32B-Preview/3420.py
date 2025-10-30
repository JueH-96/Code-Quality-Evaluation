from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Step 1: Collect indices where nums[i] == x
        indices = [i for i, num in enumerate(nums) if num == x]
        
        # Step 2: Process each query
        answer = [indices[query - 1] if query <= len(indices) else -1 for query in queries]
        
        return answer