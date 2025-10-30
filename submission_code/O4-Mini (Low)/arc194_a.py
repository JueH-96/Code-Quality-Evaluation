import sys
import heapq

def main():
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # We simulate choices by greedily keeping as large a sum as possible
    # while respecting that we only have (N - i) future operations at step i
    # to "pop" elements we've pushed.  If at step i our current stack size
    # exceeds the number of remaining steps, we are forced to issue pops
    # (each pop consumes one of the remaining steps).  When forced, we should
    # pop the smallest element in our stack (to lose as little sum as possible).
    #
    # Implementation:
    #   - We keep a min‑heap `h` of all values we've chosen to push so far.
    #   - We keep running sum `s` of those values.
    #   - At step i (0-based), we "push" A[i]:  add it to h and to s.
    #   - Then we check: our current number of pushes = len(h).
    #     The number of remaining indices (hence remaining ops) is (N - (i+1)).
    #     If len(h) > remaining_ops, we must pop exactly
    #       len(h) - remaining_ops
    #     times right now (we can actually do them lazily one by one).
    #   - Each forced pop we take the smallest from h and subtract it from s.
    #
    # At the end, s is the maximum achievable sum.
    #
    # Complexity: O(N log N).
    
    heap = []
    s = 0
    for i, x in enumerate(A):
        # push x
        heapq.heappush(heap, x)
        s += x
        # how many ops are left after this step?
        rem_ops = N - (i + 1)
        # our current stack size = len(heap)
        # if we have more items than we can ever pop in the remaining ops,
        # we must pop some now:
        while len(heap) > rem_ops:
            smallest = heapq.heappop(heap)
            s -= smallest

    print(s)

if __name__ == "__main__":
    main()