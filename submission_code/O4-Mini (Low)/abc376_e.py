import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    T = int(input())
    out_lines = []
    for _ in range(T):
        line = input().split()
        while not line:
            line = input().split()
        N, K = map(int, line)
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))

        # Pair up and sort by A ascending
        items = list(zip(A, B))
        items.sort(key=lambda x: x[0])

        # We'll maintain a max-heap of size up to K-1 for the smallest B's seen so far
        import heapq
        max_heap = []
        sum_b = 0
        ans = 10**30

        for a, b in items:
            # If we can form a set of size K with this 'a' as the max A
            # we need K-1 smallest B's from before, whose sum is sum_b
            if len(max_heap) >= K-1:
                total_b = sum_b + b
                candidate = a * total_b
                if candidate < ans:
                    ans = candidate

            # Now add this b into the heap of candidate B's
            # We keep the heap of size at most K-1, popping the largest if needed
            # Use negative values to simulate a max-heap
            heapq.heappush(max_heap, -b)
            sum_b += b
            if len(max_heap) > K-1:
                removed = -heapq.heappop(max_heap)
                sum_b -= removed

        out_lines.append(str(ans))

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()