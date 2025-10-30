import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    # Sort box sizes ascending, requirements descending
    A.sort()
    B.sort(reverse=True)
    import heapq
    heap = []
    total = 0
    j = N - 1
    for b in B:
        # push all boxes that can satisfy requirement b
        while j >= 0 and A[j] >= b:
            heapq.heappush(heap, A[j])
            j -= 1
        if not heap:
            print(-1)
            return
        # assign the cheapest available box
        total += heapq.heappop(heap)
    print(total)

if __name__ == "__main__":
    main()