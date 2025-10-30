class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # We will consider doing k = 0..(n-1) rotations in total.
        # After k rotations (cost k*x), for each chocolate type i,
        # we can choose to buy it at any rotation r where 0 <= r <= k.
        # The chocolate of type i, after r rotations, appears at index (i-r) mod n.
        # So its cost could be nums[(i-r) mod n].
        # We pick the minimum cost among r=0..k for each type i.
        # Total cost for k operations is k*x + sum( min_{r=0..k} nums[(i-r) mod n], i=0..n-1 ).
        
        # Precompute min_cost_for_each_type_up_to_k[r][i] = min_{s=0..r} nums[(i-s) mod n].
        # However, for n up to 1000, we can do a direct O(n^2) solution:
        
        INF = float('inf')
        answer = INF
        
        # We keep a running array "best[i]" which will track min_{r=0..k} nums[(i-r) mod n].
        # Initially for k=0, best[i] = nums[i].
        best = nums[:]  # for k=0
        
        # Compute cost if k=0 (no rotations)
        cost_k = sum(best)
        answer = min(answer, cost_k)  # 0*x + sum_of_costs
        
        # Now incrementally update "best" for k = 1..n-1
        for k in range(1, n):
            # For each i, update best[i] = min(best[i], nums[(i - k) mod n])
            # but we can do it in a simpler pass now that k increments by 1
            for i in range(n):
                # candidate after k-th rotation is nums[(i - k) mod n]
                candidate = nums[(i - k) % n]
                if candidate < best[i]:
                    best[i] = candidate
            
            cost_k = k * x + sum(best)
            if cost_k < answer:
                answer = cost_k
        
        return answer