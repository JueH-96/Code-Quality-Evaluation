import heapq
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1

    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N
    C = list(map(int, input[idx:idx+N]))
    idx += N

    # Sort in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    M = int(K**0.5)
    M = min(M, N)
    A_top = A[:M]
    C_top = C[:M]

    heap = []
    for a in A_top:
        for c in C_top:
            if B:  # B is non-empty as per constraints
                val = (a + c) * B[0] + a * c
                heapq.heappush(heap, (-val, a, c, 0))

    result = None
    for _ in range(K):
        current = heapq.heappop(heap)
        current_val = -current[0]
        if _ == K - 1:
            result = current_val
        a = current[1]
        c = current[2]
        b_idx = current[3]
        if b_idx + 1 < len(B):
            next_b = B[b_idx + 1]
            next_val = (a + c) * next_b + a * c
            heapq.heappush(heap, (-next_val, a, c, b_idx + 1))

    print(result)

if __name__ == "__main__":
    main()