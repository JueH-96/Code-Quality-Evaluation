import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1

    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N

    S = 300
    M = 500
    T = 1000

    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)
    C_sorted = sorted(C, reverse=True)

    A_top = A_sorted[:S]
    B_top = B_sorted[:M]
    C_top = C_sorted[:S]

    candidates = []

    for b_j in B_top:
        a_plus = [a + b_j for a in A_top]
        c_plus = [c + b_j for c in C_top]
        len_a = len(a_plus)
        len_c = len(c_plus)
        visited = [[False] * len_c for _ in range(len_a)]
        heap = []
        if len_a == 0 or len_c == 0:
            continue
        heapq.heappush(heap, (-a_plus[0] * c_plus[0], 0, 0))
        visited[0][0] = True
        current_count = 0
        products = []
        while current_count < T and heap:
            neg_product, i, j = heapq.heappop(heap)
            products.append(-neg_product)
            current_count += 1
            if i + 1 < len_a and not visited[i+1][j]:
                visited[i+1][j] = True
                heapq.heappush(heap, (-a_plus[i+1] * c_plus[j], i+1, j))
            if j + 1 < len_c and not visited[i][j+1]:
                visited[i][j+1] = True
                heapq.heappush(heap, (-a_plus[i] * c_plus[j+1], i, j+1))
        bj_sq = b_j * b_j
        for p in products:
            candidates.append(p - bj_sq)

    candidates.sort(reverse=True)
    print(candidates[K-1])

if __name__ == "__main__":
    main()