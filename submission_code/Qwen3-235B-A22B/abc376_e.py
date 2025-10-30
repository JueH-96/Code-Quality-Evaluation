import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(data[idx])
        K = int(data[idx+1])
        idx +=2
        A = list(map(int, data[idx:idx+N]))
        idx +=N
        B = list(map(int, data[idx:idx+N]))
        idx +=N
        AB = list(zip(A, B))
        AB.sort()
        heap = []
        sum_heap = 0
        ans = float('inf')
        for i in range(N):
            a_i, b_i = AB[i]
            if i >= K-1 and len(heap) >= K-1:
                current_sum = sum_heap + b_i
                product = a_i * current_sum
                if product < ans:
                    ans = product
            heapq.heappush(heap, -b_i)
            sum_heap += b_i
            while len(heap) > K-1:
                removed = -heapq.heappop(heap)
                sum_heap -= removed
        results.append(ans)
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()