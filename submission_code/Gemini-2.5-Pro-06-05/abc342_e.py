import sys
import heapq

def solve():
    """
    Solves the AtCoder train problem by finding the latest possible departure time
    from each station to reach the destination. This is accomplished using a
    Dijkstra-like algorithm operating backwards from the destination station.
    """
    # Read problem size
    try:
        N, M = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle empty input case in certain environments
        return

    # Create an adjacency list for trains, indexed by their arrival station.
    # trains_to[B] will store tuples of (l, d, k, c, A) for trains from A to B.
    trains_to = [[] for _ in range(N + 1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        trains_to[B].append((l, d, k, c, A))

    # `latest_dep[u]` will store the latest possible departure time from station `u`
    # from which it's possible to reach station N.
    
    # A large number to represent infinity. The maximum possible departure time
    # can be around 10^9 + (10^9-1)*10^9 ~= 10^18.
    # A value like 4*10^18 is a safe choice for infinity.
    INF = 4 * 10**18
    latest_dep = [-1] * (N + 1)
    
    # For the destination station N, we are already there. We can think of being able
    # to "depart" at an infinitely late time to reach it.
    latest_dep[N] = INF

    # We use a max-priority queue to explore stations in descending order of their
    # latest departure times. Python's heapq is a min-heap, so we store negative
    # time values to simulate a max-heap. The queue stores tuples of (-time, station).
    pq = [(-INF, N)]

    while pq:
        neg_time, u = heapq.heappop(pq)
        time = -neg_time

        # If we have found a better (later) path to `u` already, this path is suboptimal.
        if time < latest_dep[u]:
            continue

        # Consider all trains that arrive at station `u`.
        # For each incoming train from station `v`, check if it can improve `latest_dep[v]`.
        for l, d, k, c, v in trains_to[u]:
            # The latest we can depart from `u` is `time`. To connect to this path,
            # a train from `v` must arrive at `u` at or before `time`.
            # A train from `v` (index `i`) arrives at `u` at `l + i*d + c`.
            # We need: `l + i*d + c <= time`
            
            # If `time - c` is less than `l`, even the first train (i=0) arrives
            # too late. So, we can't take any train on this schedule.
            if time - c < l:
                continue
            
            # We need to find the largest integer `i` (train index) satisfying:
            # 1. `0 <= i < k`
            # 2. `i <= (time - c - l) / d`
            
            max_i = (time - c - l) // d
            
            # Combine the two constraints on `i`.
            max_i = min(max_i, k - 1)
            
            # The latest possible departure time from `v` using this schedule.
            dep_time_v = l + max_i * d
            
            # If this new path allows a later departure from `v`, update `latest_dep[v]`
            # and push the new state to the priority queue.
            if dep_time_v > latest_dep[v]:
                latest_dep[v] = dep_time_v
                heapq.heappush(pq, (-dep_time_v, v))

    # After the algorithm finishes, `latest_dep[i]` contains the answer for station `i`.
    # Print the results for stations 1 to N-1.
    for i in range(1, N):
        if latest_dep[i] == -1:
            print("Unreachable")
        else:
            print(latest_dep[i])

if __name__ == "__main__":
    solve()