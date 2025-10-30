def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    
    N = int(data[idx])
    idx += 1
    P = list(map(int, data[idx:idx+N]))
    idx += N
    M = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+M]))
    
    # Calculate initial inversion count
    inv_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if P[i] > P[j]:
                inv_count += 1
    
    # We will process the operations and update the permutation P
    last_k = 1
    results = []
    
    for k in A:
        # Perform operations from last_k+1 to k (inclusive)
        for op in range(last_k + 1, k + 1):
            # We only need to consider swaps from 1 to op-1
            for i in range(op - 1):
                if P[i] > P[i + 1]:
                    # Before swap, this was an inversion
                    inv_count -= 1
                    # Swap
                    P[i], P[i + 1] = P[i + 1], P[i]
                    # After swap, check if it creates a new inversion
                    if i > 0 and P[i - 1] > P[i]:
                        inv_count += 1
                    if i + 1 < N and P[i + 1] > P[i + 2]:
                        inv_count += 1
        
        # Store the result after applying operation k
        results.append(inv_count)
        last_k = k
    
    # Output all results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()