def main():
    import sys, heapq
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    k = int(data[1])
    # Read the arrays.
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+2*n]))
    C = list(map(int, data[2+2*n:2+3*n]))
    # sort in descending order.
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    # Our state is (i, j, k) (0-indexed), and since the value
    # f(i,j,k) = A[i]*B[j] + B[j]*C[k] + C[k]*A[i] is monotonic
    # (i.e. any increase of an index only decreases f),
    # we can do a best-first search starting from (0,0,0).
    #
    # We use a max-heap (implemented with negatives since Python's heapq is a min-heap)
    # and we maintain a visited set. Because n can be huge, we encode a state (i,j,k)
    # as an integer: state_id = i*(n*n) + j*n + k.
    nsq = n * n
    visited = set()
    start_state = 0  # (0,0,0) -> 0*n*n + 0*n + 0 == 0.
    visited.add(start_state)
    
    # compute function value for state (i, j, k)
    def f(i, j, k_idx):
        return A[i] * B[j] + B[j] * C[k_idx] + C[k_idx] * A[i]
    
    # initialize the heap with state (0,0,0).
    initial_val = f(0,0,0)
    heap = [(-initial_val, 0, 0, 0)]
    
    count = 0
    answer = None
    while heap:
        neg_val, i, j, k_idx = heapq.heappop(heap)
        count += 1
        val = -neg_val
        if count == k:
            answer = val
            break
        # Try to push the three neighbors (if within bounds and not visited).
        if i + 1 < n:
            new_i = i + 1
            state_id = new_i * nsq + j * n + k_idx
            if state_id not in visited:
                visited.add(state_id)
                heapq.heappush(heap, (-f(new_i, j, k_idx), new_i, j, k_idx))
        if j + 1 < n:
            new_j = j + 1
            state_id = i * nsq + new_j * n + k_idx
            if state_id not in visited:
                visited.add(state_id)
                heapq.heappush(heap, (-f(i, new_j, k_idx), i, new_j, k_idx))
        if k_idx + 1 < n:
            new_k = k_idx + 1
            state_id = i * nsq + j * n + new_k
            if state_id not in visited:
                visited.add(state_id)
                heapq.heappush(heap, (-f(i, j, new_k), i, j, new_k))
                
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()