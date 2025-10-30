import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        def can_achieve(target_sum):
            current_sum = sum(nums)
            required_reduction = current_sum - target_sum
            if required_reduction <= 0:
                return True

            potential_reductions = []
            for i in range(len(nums)):
                # Potential reduction from Operation 1
                reduction1 = nums[i] - math.ceil(nums[i] / 2)
                potential_reductions.append((-reduction1, 1, i))  # Negate for max-heap

                # Potential reduction from Operation 2
                if nums[i] >= k:
                    potential_reductions.append((-k, 2, i))

            import heapq
            heapq.heapify(potential_reductions)

            achieved_reduction = 0
            op1_used = 0
            op2_used = 0
            op1_applied = [False] * len(nums)
            op2_applied = [False] * len(nums)

            while potential_reductions and achieved_reduction < required_reduction:
                reduction, op_type, index = heapq.heappop(potential_reductions)
                reduction = -reduction

                if op_type == 1 and op1_used < op1 and not op1_applied[index]:
                    achieved_reduction += reduction
                    op1_used += 1
                    op1_applied[index] = True
                elif op_type == 2 and op2_used < op2 and not op2_applied[index]:
                    achieved_reduction += reduction
                    op2_used += 1
                    op2_applied[index] = True

            return achieved_reduction >= required_reduction

        low = 0
        high = sum(nums)
        ans = high

        while low <= high:
            mid = (low + high) // 2
            if can_achieve(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans