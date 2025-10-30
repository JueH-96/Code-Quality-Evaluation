import heapq
from typing import List
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        diffs = []
        for x in nums:
            diff1 = x - (x + 1) // 2
            diff2 = x - max(0, x - k)
            diffs.append((diff1, diff2))

        heap = []
        for i in range(n):
            d1, d2 = diffs[i]
            if d1 > 0 and d2 > 0:
                if d1 > d2:
                    heapq.heappush(heap, (-d1, i, 1))
                    heapq.heappush(heap, (-(d2 - (max(0, (nums[i] + 1) // 2 - k) - (nums[i] + 1) // 2)), i, 2))
                else:
                    heapq.heappush(heap, (-d2, i, 2))
                    heapq.heappush(heap, (-(d1 - (max(0, nums[i] - k) - (max(0, nums[i] - k) + 1) // 2)), i, 1))
            elif d1 > 0:
                heapq.heappush(heap, (-d1, i, 1))
            elif d2 > 0:
                heapq.heappush(heap, (-d2, i, 2))

        total_ops = op1 + op2
        total_sum = sum(nums)

        while total_ops > 0 and heap:
            diff, i, op_type = heapq.heappop(heap)
            diff = -diff
            
            if op_type == 1 and op1 > 0:
                total_sum -= diff
                nums[i] = (nums[i] + 1) // 2
                op1 -= 1
                total_ops -= 1

                d1 = nums[i] - (nums[i] + 1) // 2
                d2 = nums[i] - max(0, nums[i] - k)
                if d1 > 0 and d2 > 0:
                    if d1 > d2:
                        heapq.heappush(heap, (-d1, i, 1))
                        heapq.heappush(heap, (-(d2 - (max(0, (nums[i] + 1) // 2 - k) - (nums[i] + 1) // 2)), i, 2))
                    else:
                        heapq.heappush(heap, (-d2, i, 2))
                        heapq.heappush(heap, (-(d1 - (max(0, nums[i] - k) - (max(0, nums[i] - k) + 1) // 2)), i, 1))
                elif d1 > 0:
                    heapq.heappush(heap, (-d1, i, 1))
                elif d2 > 0:
                    heapq.heappush(heap, (-d2, i, 2))

            elif op_type == 2 and op2 > 0:
                total_sum -= diff
                nums[i] = max(0, nums[i] - k)
                op2 -= 1
                total_ops -= 1

                d1 = nums[i] - (nums[i] + 1) // 2
                d2 = nums[i] - max(0, nums[i] - k)
                if d1 > 0 and d2 > 0:
                    if d1 > d2:
                        heapq.heappush(heap, (-d1, i, 1))
                        heapq.heappush(heap, (-(d2 - (max(0, (nums[i] + 1) // 2 - k) - (nums[i] + 1) // 2)), i, 2))
                    else:
                        heapq.heappush(heap, (-d2, i, 2))
                        heapq.heappush(heap, (-(d1 - (max(0, nums[i] - k) - (max(0, nums[i] - k) + 1) // 2)), i, 1))
                elif d1 > 0:
                    heapq.heappush(heap, (-d1, i, 1))
                elif d2 > 0:
                    heapq.heappush(heap, (-d2, i, 2))

        return total_sum