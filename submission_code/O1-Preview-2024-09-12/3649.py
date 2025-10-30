class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        import heapq
        from math import ceil

        n = len(strength)
        max_X = 1 + K * n  # Maximum possible value of X
        max_states = (1 << n) * (max_X + 1)

        # Initialize distance dictionary
        dist = {}
        import collections
        dist = collections.defaultdict(lambda: float('inf'))

        # Initial state: all locks remaining, X=1, time=0
        initial_state = ( (1 << n) -1, 1)
        dist[initial_state] = 0
        heap = [(0, ( (1 << n) -1, 1))]  # (time, (locks_remaining_bitmap, X))

        while heap:
            current_time, (locks_remaining, X) = heapq.heappop(heap)

            # If all locks are broken
            if locks_remaining == 0:
                return current_time  # Minimum time required

            # If we have already found a better way to this state
            if dist[(locks_remaining, X)] < current_time:
                continue

            # For each lock still remaining
            for i in range(n):
                if (locks_remaining >> i) & 1:
                    strength_i = strength[i]
                    t_i = ceil(strength_i / X)
                    next_time = current_time + t_i
                    next_locks_remaining = locks_remaining & (~(1 << i))
                    next_X = X + K
                    next_state = (next_locks_remaining, next_X)

                    if dist[next_state] > next_time:
                        dist[next_state] = next_time
                        heapq.heappush(heap, (next_time, next_state))

        # If we get here, all locks cannot be broken (should not happen with given constraints)
        return -1