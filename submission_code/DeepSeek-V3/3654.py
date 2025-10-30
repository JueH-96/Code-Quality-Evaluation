from typing import List
import heapq

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Create a max-heap based on the potential reduction from operation 1
        heap = []
        for num in nums:
            # Calculate the reduction if operation 1 is applied
            reduction_op1 = num - (num + 1) // 2
            # Calculate the reduction if operation 2 is applied
            reduction_op2 = k if num >= k else 0
            # Push the maximum possible reduction for each number
            max_reduction = max(reduction_op1, reduction_op2)
            heapq.heappush(heap, (-max_reduction, num, reduction_op1, reduction_op2))
        
        # Initialize the sum of the original array
        total_sum = sum(nums)
        
        # Perform the operations
        while (op1 > 0 or op2 > 0) and heap:
            neg_reduction, num, red_op1, red_op2 = heapq.heappop(heap)
            reduction = -neg_reduction
            if red_op1 == reduction and op1 > 0:
                # Apply operation 1
                total_sum -= red_op1
                op1 -= 1
            elif red_op2 == reduction and op2 > 0:
                # Apply operation 2
                total_sum -= red_op2
                op2 -= 1
            else:
                # No operation can be applied, continue
                continue
        
        return total_sum