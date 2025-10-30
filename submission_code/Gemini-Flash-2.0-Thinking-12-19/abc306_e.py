import heapq

def solve():
    n, k, q = map(int, input().split())
    a = [0] * n
    for _ in range(q):
        x, y = map(int, input().split())
        a[x-1] = y
        min_heap = []
        current_sum = 0
        for val in a:
            if len(min_heap) < k:
                heapq.heappush(min_heap, val)
            else:
                if val > min_heap[0]:
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap, val)
        current_sum = sum(min_heap)
        print(current_sum)

if __name__ == '__main__':
    solve()