import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = map(int, input[ptr:ptr+2])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        if K == 1:
            min_val = min(a * b for a, b in zip(A, B))
            print(min_val)
            continue
        # Proceed with the heap approach
        pairs = sorted(zip(A, B), key=lambda x: x[0])
        A_sorted = [x[0] for x in pairs]
        B_sorted = [x[1] for x in pairs]
        heap = []
        current_sum = 0
        ans = float('inf')
        for i in range(N):
            current_A = A_sorted[i]
            current_B = B_sorted[i]
            if i >= K - 1:
                candidate = current_A * (current_sum + current_B)
                if candidate < ans:
                    ans = candidate
            # Update the heap
            if len(heap) < K - 1:
                heapq.heappush(heap, -current_B)
                current_sum += current_B
            else:
                if current_B < -heap[0]:
                    removed = -heapq.heappop(heap)
                    current_sum -= removed
                    current_sum += current_B
                    heapq.heappush(heap, -current_B)
        print(ans)

if __name__ == "__main__":
    main()