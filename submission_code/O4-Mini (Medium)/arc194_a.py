import sys
import threading
import heapq

def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # We maintain a min-heap of the values we have "pushed" so far.
    # At each step i we must choose push or pop.
    #  Let rem = N - i be the number of steps left after i.
    #  Let sz = current stack size.
    #  We can never allow sz > rem + final_size_min, but final_size_min = 0
    #  so we must keep sz <= rem, else we would be forced to pop more than we can.
    #
    # Greedy rule:
    #   Always push A[i].
    #   After pushing, if sz > rem, we must immediately pop once.
    #   To pop so as to do least damage, we remove the smallest element we have ever pushed (heap[0]).
    #
    # At the end, the heap contains exactly the elements that survive in S.
    # Their sum is the answer.
    #
    heap = []
    total = 0
    sz = 0
    for i, x in enumerate(A):
        # push
        heapq.heappush(heap, x)
        total += x
        sz += 1
        rem = N - (i+1)
        # if we ever have more on stack than steps left, we pop one
        if sz > rem:
            # pop the worst (smallest) element
            worst = heapq.heappop(heap)
            total -= worst
            sz -= 1

    # At the end, sz should be <= 0 + rem(=0), so sz==0, but we
    # actually accumulate in total only those that survive.
    print(total)

if __name__ == "__main__":
    main()