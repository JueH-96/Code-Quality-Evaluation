import heapq
from collections import defaultdict
from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)

        def check(k):
            if k == 0:
                return all(x == 0 for x in nums)

            # remaining_needed[i] tracks the total reduction that must be applied to index i
            # by queries j < k with l_j <= i <= r_j.
            # Processing from right to left, remaining_needed[i] initially is nums[i].
            # Reductions from queries ending >= i are accounted for via the carry-over.
            remaining_needed = list(nums)

            # Store queries by their right endpoint for efficient lookup.
            # queries_by_r[r] will store (val, l) for queries j < k with r_j = r.
            queries_by_r = defaultdict(list)
            for j in range(k):
                l, r, val = queries[j]
                queries_by_r[r].append((val, l))

            # active_offers heap stores (val, l) for queries j < k that cover the current index i or are to its left (l_j <= i) AND whose right endpoint r_j >= i.
            # A query j=(l_j, r_j, val_j) is active at index i if l_j <= i <= r_j.
            # As we process i from n-1 down to 0:
            # - Queries with r_j = i become active for indices <= i. Add them to the heap.
            # - Queries with r_j > i that were active at i+1 remain active at i if l_j <= i.
            # - Queries with l_j > i cannot cover index i or anything left of i. Remove them from consideration (prune).
            # The heap `active_offers` stores offers (val, l) where r_j >= current_i.
            # We maintain `current_sum_val` which is the sum of `val` for offers `(val, l)` in `active_offers` where `l <= current_i`.

            active_offers = [] # Min-heap of (val, l)
            current_sum_val = 0 # Sum of val for offers (val, l) in active_offers where l <= current_i

            for i in range(n - 1, -1, i - 1):
                # Add offers ending exactly at `i`. These become available for indices <= i.
                # Queries j with r_j = i are added.
                for val, l in queries_by_r[i]:
                    # This offer (val, l) is available for indices m in [l, i].
                    # Add it to the pool of active offers for indices <= i.
                    heapq.heappush(active_offers, (val, l))
                    current_sum_val += val # Add its contribution to the sum of active offers (potentially including l > i)

                # Prune offers from the heap that start right of `i`.
                # Offers (val, l) in the heap that have l > i cannot cover index i or indices < i.
                # We filter these out and remove their contribution from `current_sum_val`.
                while active_offers and active_offers[0][1] > i:
                    val, l = heapq.heappop(active_offers)
                    current_sum_val -= val # Remove its contribution from the sum


                # `current_sum_val` now represents the total reduction capacity from queries
                # j < k with l_j <= i <= r_j.
                # This total capacity is available for index `i` and indices < `i` within their ranges.
                # The total reduction *applied* at index `i` must be exactly `remaining_needed[i]`.
                # This reduction is drawn from the `current_sum_val` pool.
                # If the total available `current_sum_val` is less than the amount needed at `i`, it's impossible.
                if current_sum_val < remaining_needed[i]:
                    return False

                # We apply `remaining_needed[i]` reduction at index `i`.
                # The remaining reduction capacity `current_sum_val - remaining_needed[i]` is carried over
                # to indices < i. This carried amount reduces the requirements for indices < i.
                # The total reduction carried over that is available for indices < i is `excess = current_sum_val - remaining_needed[i]`.
                # This excess reduces the requirement for `remaining_needed[i-1]`.
                # This is because `remaining_needed[i-1]` represents the requirement for index `i-1` from queries covering `i-1` or left, *not yet accounted for by carry from right*.
                # The carry from `i` *is* accounted for by reducing `remaining_needed[i-1]`.
                # This entire excess `current_sum_val - remaining_needed[i]` is available for *any* index `m < i` that is covered by the offers contributing to the excess.
                # Thus, the total requirement at `i-1` is reduced by this entire excess amount.
                if i > 0:
                    remaining_needed[i - 1] -= (current_sum_val - remaining_needed[i])

            # If the loop finishes for all indices and no impossibility was detected, it's possible.
            return True

        # Binary search for the minimum k.
        low = 0
        high = m
        ans = -1

        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                high = mid - 1 # Try smaller k
            else:
                low = mid + 1 # Need larger k

        return ans