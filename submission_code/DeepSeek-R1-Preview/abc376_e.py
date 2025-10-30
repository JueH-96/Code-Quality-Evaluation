import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr+1])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        if K == 1:
            min_val = min(a * b for a, b in zip(A, B))
        else:
            pairs = list(zip(A, B))
            pairs.sort()
            A_sorted, B_sorted = zip(*pairs)
            A_sorted = list(A_sorted)
            B_sorted = list(B_sorted)
            prefix_sum = [0] * N
            heap = []
            current_sum = 0
            for j in range(N):
                bj = B_sorted[j]
                heapq.heappush(heap, -bj)
                current_sum += bj
                if len(heap) > K - 1:
                    popped = -heapq.heappop(heap)
                    current_sum -= popped
                prefix_sum[j] = current_sum
            min_val = float('inf')
            for i in range(K-1, N):
                if i - 1 >= 0:
                    sum_b = prefix_sum[i-1] + B_sorted[i]
                else:
                    sum_b = B_sorted[i]
                product = A_sorted[i] * sum_b
                if product < min_val:
                    min_val = product
        print(min_val)

if __name__ == '__main__':
    main()