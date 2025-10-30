def solve():
    import sys
    import heapq
    
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    
    # Prepare queries
    idx = 2 + N
    queries = []
    for _ in range(Q):
        i_k = int(input_data[idx]); x_k = int(input_data[idx + 1])
        idx += 2
        queries.append((i_k, x_k))
    
    # Frequency array up to N+1 (we only need to track 0..N because mex can't exceed N+1)
    freq = [0] * (N + 2)
    
    # Initialize frequency for elements (only if they are <= N)
    for val in A:
        if val <= N:
            freq[val] += 1
    
    # Initialize heap of zero-frequency candidates
    zero_candidates = []
    for i in range(N + 2):
        if freq[i] == 0:
            heapq.heappush(zero_candidates, i)
    
    # Process queries
    out = []
    for (i_k, x_k) in queries:
        # 1. Remove the old value
        old_val = A[i_k - 1]
        if old_val <= N:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                heapq.heappush(zero_candidates, old_val)
        
        # 2. Insert the new value
        A[i_k - 1] = x_k
        if x_k <= N:
            if freq[x_k] == 0:
                heapq.heappush(zero_candidates, x_k)
            freq[x_k] += 1
        
        # 3. Find current mex (top of the heap with freq==0)
        while freq[zero_candidates[0]] != 0:
            heapq.heappop(zero_candidates)
        out.append(str(zero_candidates[0]))
    
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()