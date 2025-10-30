# YOUR CODE HERE
import sys, heapq

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    L = int(data[idx]); idx +=1
    a = []
    for _ in range(N):
        a.append(int(data[idx]))
        idx +=1
    b = []
    for _ in range(M):
        b.append(int(data[idx]))
        idx +=1
    forbidden_input = []
    for _ in range(L):
        c = int(data[idx]); idx +=1
        d = int(data[idx]); idx +=1
        forbidden_input.append( (c, d) )
    # Sort main dishes descending with original indices
    sorted_main = sorted( [(a[i], i+1) for i in range(N)], key=lambda x: -x[0])
    # Sort side dishes descending with original indices
    sorted_side = sorted( [(b[j], j+1) for j in range(M)], key=lambda x: -x[0])
    # Create mapping from original to sorted indices
    orig_to_sorted_a = [0]*(N+1)
    for sorted_i, (val, orig_i) in enumerate(sorted_main):
        orig_to_sorted_a[orig_i] = sorted_i
    orig_to_sorted_b = [0]*(M+1)
    for sorted_j, (val, orig_j) in enumerate(sorted_side):
        orig_to_sorted_b[orig_j] = sorted_j
    # Create forbidden set in sorted indices
    forbidden_set = set()
    for c, d in forbidden_input:
        si = orig_to_sorted_a[c]
        sj = orig_to_sorted_b[d]
        forbidden_set.add( si * M + sj )
    # Initialize heap and visited
    heap = []
    initial_sum = sorted_main[0][0] + sorted_side[0][0]
    heapq.heappush(heap, (-initial_sum, 0, 0))
    visited = set()
    visited.add(0 * M + 0)
    while heap:
        neg_sum, i, j = heapq.heappop(heap)
        current_sum = -neg_sum
        key = i * M + j
        if key not in forbidden_set:
            print(current_sum)
            return
        # Push (i+1, j)
        if i +1 < N:
            ni, nj = i+1, j
            nkey = ni * M + nj
            if nkey not in visited:
                new_sum = sorted_main[ni][0] + sorted_side[nj][0]
                heapq.heappush(heap, (-new_sum, ni, nj))
                visited.add(nkey)
        # Push (i, j+1)
        if j +1 < M:
            ni, nj = i, j+1
            nkey = ni * M + nj
            if nkey not in visited:
                new_sum = sorted_main[ni][0] + sorted_side[nj][0]
                heapq.heappush(heap, (-new_sum, ni, nj))
                visited.add(nkey)

if __name__ == "__main__":
    main()