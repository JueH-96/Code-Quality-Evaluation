import heapq
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        current_nums = list(nums)
        n = len(nums)
        op1_used_count = 0
        op2_used_count = 0
        index_op1_used = [0] * n
        index_op2_used = [0] * n
        pq = []

        for i in range(n):
            reduction_op1 = current_nums[i] - (current_nums[i] + 1) // 2
            if reduction_op1 > 0:
                heapq.heappush(pq, (-reduction_op1, i, 1))
            reduction_op2 = k if current_nums[i] >= k else 0
            if reduction_op2 > 0:
                heapq.heappush(pq, (-reduction_op2, i, 2))

        while op1_used_count < op1 or op2_used_count < op2:
            if not pq:
                break
            neg_reduction, index, op_type = heapq.heappop(pq)
            reduction = -neg_reduction

            if op_type == 1:
                if op1_used_count < op1 and index_op1_used[index] == 0:
                    current_nums[index] = (current_nums[index] + 1) // 2
                    op1_used_count += 1
                    index_op1_used[index] = 1
                    reduction_op1 = current_nums[index] - (current_nums[index] + 1) // 2
                    if reduction_op1 > 0 and index_op1_used[index] == 0:
                        heapq.heappush(pq, (-reduction_op1, index, 1))
                    reduction_op2 = k if current_nums[index] >= k else 0
                    if reduction_op2 > 0 and index_op2_used[index] == 0:
                        heapq.heappush(pq, (-reduction_op2, index, 2))
            elif op_type == 2:
                if op2_used_count < op2 and index_op2_used[index] == 0:
                    if current_nums[index] >= k:
                        current_nums[index] -= k
                        op2_used_count += 1
                        index_op2_used[index] = 1
                        reduction_op1 = current_nums[index] - (current_nums[index] + 1) // 2
                        if reduction_op1 > 0 and index_op1_used[index] == 0:
                            heapq.heappush(pq, (-reduction_op1, index, 1))
                        reduction_op2 = k if current_nums[index] >= k else 0
                        if reduction_op2 > 0 and index_op2_used[index] == 0:
                            heapq.heappush(pq, (-reduction_op2, index, 2))
                            
        return sum(current_nums)