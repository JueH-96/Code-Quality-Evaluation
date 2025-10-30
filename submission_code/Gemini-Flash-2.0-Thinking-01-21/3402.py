from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        max_val = max(nums)
        diffs = [max_val - x for x in nums]
        total_increase = sum(diffs)
        if 2 * cost1 <= cost2:
            return (total_increase * cost1) % (10**9 + 7)
        else:
            op2_count = 0
            op1_count = 0
            current_diffs = list(diffs)
            while True:
                positive_indices = []
                for i in range(len(current_diffs)):
                    if current_diffs[i] > 0:
                        positive_indices.append(i)
                if len(positive_indices) < 2:
                    break
                index1 = positive_indices[0]
                index2 = positive_indices[1]
                current_diffs[index1] -= 1
                current_diffs[index2] -= 1
                op2_count += 1
            for diff in current_diffs:
                if diff > 0:
                    op1_count += diff
            return (op2_count * cost2 + op1_count * cost1) % (10**9 + 7)