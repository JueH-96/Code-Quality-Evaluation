import sys
import itertools

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        N, Q = map(int, input().split())
        x = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential errors during input reading
        return

    # Step 1: Simulate the queries to get the sequence of set sizes.
    S = set()
    sizes = [0] * Q
    for i in range(Q):
        val = x[i]
        if val in S:
            S.remove(val)
        else:
            S.add(val)
        sizes[i] = len(S)

    # Step 2: Compute prefix sums of the sizes array using itertools.accumulate
    # for a clean and efficient implementation. prefix_sizes[i] will be sum(sizes[:i]).
    prefix_sizes = [0] + list(itertools.accumulate(sizes))

    # Step 3: Group query indices by the number `x_i`.
    # `events[k]` will list the 0-indexed times when `k` was queried.
    events = [[] for _ in range(N + 1)]
    for i, val in enumerate(x):
        events[val].append(i)

    # Step 4: Calculate the final values for the sequence A.
    A = [0] * N
    for k in range(1, N + 1):
        k_times = events[k]
        
        # An element `k` is in the set S during intervals defined by pairs of its
        # query times (e.g., from the 1st to 2nd time, 3rd to 4th, etc.).
        for i in range(0, len(k_times), 2):
            start_time = k_times[i]
            
            # If there's an odd number of queries for `k`, it remains in the set
            # from its last query time until the end of all queries.
            if i + 1 < len(k_times):
                end_time = k_times[i + 1]
            else:
                end_time = Q
            
            # The total value added to A_k is the sum of |S| over the intervals
            # it was in the set. We use prefix sums for efficient calculation:
            # sum(sizes[start_time:end_time]) = prefix_sizes[end_time] - prefix_sizes[start_time]
            contribution = prefix_sizes[end_time] - prefix_sizes[start_time]
            A[k - 1] += contribution
            
    # Step 5: Print the final sequence A to stdout.
    print(*A)

if __name__ == "__main__":
    main()