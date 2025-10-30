import heapq

def main():
    import sys
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
    
    # Sort A, B, C in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    candidates = []
    
    # Adjusted parameters to cover more cases
    max_j = min(N, 200)
    T = min(5000, N * N)
    
    for j in range(max_j):
        bj = B[j]
        visited = set()
        heap = []
        if N == 0:
            break
        first_product = (A[0] + bj) * (C[0] + bj)
        heapq.heappush(heap, (-first_product, 0, 0))
        visited.add((0, 0))
        current_terms = []
        # Collect up to T terms or until heap is exhausted
        while len(current_terms) < T:
            if not heap:
                break
            neg_prod, i, k = heapq.heappop(heap)
            product = -neg_prod
            current_term = product - bj * bj
            current_terms.append(current_term)
            # Push next states if not visited
            if (i + 1 < N) and ((i+1, k) not in visited):
                p = (A[i+1] + bj) * (C[k] + bj)
                heapq.heappush(heap, (-p, i+1, k))
                visited.add((i+1, k))
            if (k + 1 < N) and ((i, k+1) not in visited):
                p = (A[i] + bj) * (C[k+1] + bj)
                heapq.heappush(heap, (-p, i, k+1))
                visited.add((i, k+1))
        candidates.extend(current_terms)
    
    # Sort all collected candidates
    candidates.sort(reverse=True)
    print(candidates[K-1])

if __name__ == '__main__':
    main()