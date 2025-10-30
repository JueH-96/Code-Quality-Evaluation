from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # best[i] will track the minimum cost we've seen so far
        # for buying type i over all rotations up to the current k
        best = nums[:]  
        
        # initial cost with 0 operations
        curr_sum = sum(best)
        answer = curr_sum
        
        # try performing k rotations (1 <= k < n)
        # after k rotations, for each type i, we can choose to have bought it
        # at any rotation <= k, so we keep the running minimum
        for k in range(1, n):
            curr_sum = 0
            for i in range(n):
                # the chocolate of type i after k rotations was originally at index (i - k) mod n
                j = (i - k) % n
                # update best cost for type i
                if nums[j] < best[i]:
                    best[i] = nums[j]
                curr_sum += best[i]
            # total cost = cost of k operations + sum of best purchase costs
            total_cost = curr_sum + k * x
            if total_cost < answer:
                answer = total_cost
        
        return answer