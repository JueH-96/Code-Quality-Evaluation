import heapq
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    result = [0] * N
    for i in range(N):
        current_left = i
        current_right = i
        total = A[i]
        heap = []
        if i - 1 >= 0 and A[i-1] < total:
            heapq.heappush(heap, (A[i-1], 0))
        if i + 1 < N and A[i+1] < total:
            heapq.heappush(heap, (A[i+1], 1))
        while heap:
            val, direction = heapq.heappop(heap)
            if direction == 0:
                new_pos = current_left - 1
                total += A[new_pos]
                current_left = new_pos
                if new_pos - 1 >= 0:
                    if A[new_pos - 1] < total:
                        heapq.heappush(heap, (A[new_pos - 1], 0))
            else:
                new_pos = current_right + 1
                total += A[new_pos]
                current_right = new_pos
                if new_pos + 1 < N:
                    if A[new_pos + 1] < total:
                        heapq.heappush(heap, (A[new_pos + 1], 1))
        result[i] = total
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()