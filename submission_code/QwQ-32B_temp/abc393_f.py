import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    
    queries = []
    for i in range(Q):
        R = int(input[idx])
        idx += 1
        X = int(input[idx])
        idx += 1
        queries.append((R, X, i))
    
    # Sort queries by R
    queries.sort(key=lambda x: x[0])
    
    ans = [0] * Q
    tails = []
    j = 0  # pointer to current query
    
    for i in range(1, N+1):
        x = A[i-1]
        # Update tails array
        idx_bisect = bisect.bisect_left(tails, x)
        if idx_bisect == len(tails):
            tails.append(x)
        else:
            tails[idx_bisect] = x
        
        # Process all queries with R == i
        while j < Q and queries[j][0] == i:
            R, X, original_idx = queries[j]
            # Find the answer using bisect_right
            ans_idx = bisect.bisect_right(tails, X)
            ans[original_idx] = ans_idx
            j += 1
    
    for a in ans:
        print(a)
    
if __name__ == "__main__":
    main()