from typing import List
import heapq

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Create a max-heap based on the potential reduction from Operation 1
        heap = []
        for num in nums:
            # Calculate the reduction if Operation 1 is applied
            reduction_op1 = num - (num + 1) // 2
            # Calculate the reduction if Operation 2 is applied
            reduction_op2 = k if num >= k else 0
            # Push the maximum possible reduction for each element
            heapq.heappush(heap, (-max(reduction_op1, reduction_op2), reduction_op1, reduction_op2))
        
        # Initialize the sum of the original array
        total_sum = sum(nums)
        
        # Perform the operations
        while (op1 > 0 or op2 > 0) and heap:
            max_reduction, red_op1, red_op2 = heapq.heappop(heap)
            max_reduction = -max_reduction
            if max_reduction == 0:
                continue
            if red_op1 >= red_op2 and op1 > 0:
                total_sum -= red_op1
                op1 -= 1
            elif red_op2 > red_op1 and op2 > 0:
                total_sum -= red_op2
                op2 -= 1
            else:
                if op1 > 0:
                    total_sum -= red_op1
                    op1 -= 1
                elif op2 > 0:
                    total_sum -= red_op2
                    op2 -= 1
        
        return total_sum