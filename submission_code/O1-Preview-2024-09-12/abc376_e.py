# YOUR CODE HERE
import sys
import threading
def main():
    import heapq
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    T = int(sys.stdin.readline())
    total_N = 0
    for _ in range(T):
        N, K = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
        total_N += N
        
        AB = list(zip(A, B))
        AB.sort()
        heap = []
        sum_B = 0
        ans = float('inf')
        for a, b in AB:
            heapq.heappush(heap, -b)  # Max-heap using negative values
            sum_B += b
            if len(heap) > K:
                removed_b = -heapq.heappop(heap)
                sum_B -= removed_b
            if len(heap) == K:
                E = a * sum_B
                if E < ans:
                    ans = E
        print(ans)
                
threading.Thread(target=main).start()