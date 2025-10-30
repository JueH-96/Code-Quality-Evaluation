import heapq
from typing import List, Tuple

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Initial fill of the queue with up to first three indices
        init_q = tuple(range(min(3, n)))
        init_p = len(init_q)
        # The target state is when we've consumed all original indices (p == n)
        # and the queue of remaining is empty.
        target = (n, ())

        # Dijkstra's algorithm over states (p, q)
        # where p is the next index in nums to be enqueued,
        # and q is a tuple of current buffered indices (up to 3).
        dist = {}
        start = (init_p, init_q)
        dist[start] = 0
        heap = [(0, start)]

        while heap:
            cur_cost, (p, q) = heapq.heappop(heap)
            if cur_cost > dist[(p, q)]:
                continue
            # Check for goal
            if (p, q) == target:
                return cur_cost

            # If queue has fewer than 3 and we can read more, auto-enqueue
            if len(q) < 3 and p < n:
                nxt = (p+1, q + (p,))
                if nxt not in dist or cur_cost < dist[nxt]:
                    dist[nxt] = cur_cost
                    heapq.heappush(heap, (cur_cost, nxt))
                continue

            # If fewer than 3 elements remain overall, we can remove them all at once
            if len(q) < 3 and p >= n and q:
                cost = max(nums[idx] for idx in q)
                nxt = (p, ())
                new_cost = cur_cost + cost
                if nxt not in dist or new_cost < dist[nxt]:
                    dist[nxt] = new_cost
                    heapq.heappush(heap, (new_cost, nxt))
                continue

            # Otherwise, we have at least 3 in queue: choose any two of the first three to remove
            # Enumerate all pairs among q[0], q[1], q[2]
            first_three = q[:3]
            for i in range(3):
                for j in range(i+1, 3):
                    idx1, idx2 = first_three[i], first_three[j]
                    cost = max(nums[idx1], nums[idx2])
                    # Build new queue after removing those two
                    new_q_list = [x for x in q if x not in (idx1, idx2)]
                    new_p = p
                    # Refill up to 3 from the remaining original indices
                    while len(new_q_list) < 3 and new_p < n:
                        new_q_list.append(new_p)
                        new_p += 1
                    new_q = tuple(new_q_list)
                    new_state = (new_p, new_q)
                    new_cost = cur_cost + cost
                    if new_state not in dist or new_cost < dist[new_state]:
                        dist[new_state] = new_cost
                        heapq.heappush(heap, (new_cost, new_state))

        # Fallback (should never happen for valid input)
        return 0