import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(lambda x: int(x) - 1, data[2:N+2]))
    A = list(map(int, data[N+2:]))
    
    # Create a list of lists to store the results of each operation
    results = [A]
    
    # Perform the operation up to 2*N times to detect cycles
    for _ in range(min(K, 2*N)):
        B = [A[X[i]] for i in range(N)]
        if B in results:
            break
        results.append(B)
        A = B
    
    # If K is larger than 2*N, use the cycle to find the result
    if K >= 2*N:
        cycle_start = results.index(B)
        cycle_length = len(results) - cycle_start
        K %= cycle_length
        A = results[cycle_start + K]
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()