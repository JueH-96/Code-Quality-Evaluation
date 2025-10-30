# YOUR CODE HERE
import sys
import heapq

def solve():
    """
    Solves the slime synthesis problem by processing slime sizes in ascending order
    using a min-heap to find the minimum possible number of slimes.
    """
    
    # Fast I/O
    readline = sys.stdin.readline
    
    try:
        N_str = readline()
        if not N_str:
            print(0)
            return
        N = int(N_str)
    except (ValueError, IndexError):
        print(0)
        return

    # slimes: a dictionary to map size -> count
    slimes = {}
    for _ in range(N):
        try:
            s, c = map(int, readline().split())
            # Per the problem, initial sizes S_i are all different.
            slimes[s] = c
        except (ValueError, IndexError):
            # Gracefully handle empty or malformed lines
            continue

    if not slimes:
        print(0)
        return

    # A min-heap (priority queue) to process slime sizes in ascending order.
    # Initialize it with all the initial slime sizes.
    pq = list(slimes.keys())
    heapq.heapify(pq)

    # This variable will accumulate the final number of slimes.
    final_slime_count = 0

    while pq:
        # Get the smallest available slime size from the priority queue.
        size = heapq.heappop(pq)

        # Get the current number of slimes of this size.
        # This count includes initial slimes and any slimes created from synthesis.
        count = slimes.get(size, 0)

        if count == 0:
            continue

        # Slimes that cannot be paired up (the remainder) contribute to the final count.
        final_slime_count += count % 2

        # Slimes that can be paired up are used for synthesis.
        num_new_slimes = count // 2

        if num_new_slimes > 0:
            new_size = 2 * size
            
            # Get the existing count of slimes at the new, doubled size.
            old_count_at_new_size = slimes.get(new_size, 0)

            # Add the newly created slimes to the count for the new size.
            slimes[new_size] = old_count_at_new_size + num_new_slimes

            # If this new size was not present before (i.e., its count was 0),
            # it must be added to the priority queue to be processed in the future.
            if old_count_at_new_size == 0:
                heapq.heappush(pq, new_size)

    print(final_slime_count)

solve()