import sys
import heapq

def solve():
    """
    Solves a single test case for the problem.
    """
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, K = map(int, line.split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # Combine A and B into pairs (A_i, B_i) and sort by A_i
    pairs = sorted(zip(A, B))

    min_prod = float('inf')
    
    # small_H stores the K-1 smallest B's from the pairs processed so far.
    # It's a max-heap to easily find the largest among these smallest B's.
    # We use negative values to simulate a max-heap with Python's min-heap (heapq).
    small_H = []
    sum_small = 0
    
    # large_H stores other B's that are not among the K-1 smallest.
    # It's a min-heap.
    large_H = []

    # Iterate through the sorted pairs
    for i in range(N):
        a, b = pairs[i]
        
        # A valid set of size K can be formed if we have processed at least K-1 pairs
        # before this one. The first time this is possible is at index i = K-1.
        if i >= K - 1:
            # At this stage, `small_H` contains the K-1 smallest B's from the pool
            # {B_0, ..., B_{i-1}}, and `sum_small` is their sum.
            # If K=1, small_H is empty and sum_small is 0.
            
            # The current candidate set S has P_i as its max-A element.
            # The sum of B's is `b` (from P_i) plus the K-1 smallest B's from previous pairs.
            current_sum_b = b + sum_small
            current_prod = a * current_sum_b
            min_prod = min(min_prod, current_prod)

        # Update the heaps with the current B value `b`. This prepares the heaps for
        # the next iteration, where P_i becomes part of the pool of "other" elements.
        if K == 1:
            # If K=1, the sum of B's is just B_i. sum_small is always 0. No heaps needed.
            continue

        if len(small_H) < K - 1:
            heapq.heappush(small_H, -b)
            sum_small += b
        # If b is smaller than the largest element in small_H, it belongs in small_H.
        elif b < -small_H[0]:
            # The largest element in small_H is evicted and moves to large_H.
            out_val = -heapq.heapreplace(small_H, -b)
            sum_small += b - out_val
            heapq.heappush(large_H, out_val)
        else:
            # Otherwise, b is larger or equal, so it goes to large_H.
            heapq.heappush(large_H, b)
            
    print(min_prod)

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        num_test_cases_str = sys.stdin.readline()
        if not num_test_cases_str: return
        num_test_cases = int(num_test_cases_str)
        for _ in range(num_test_cases):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()