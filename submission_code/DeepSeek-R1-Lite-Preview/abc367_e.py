import sys
import array

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    # Convert X to 0-based indexing
    X = [x - 1 for x in X]
    
    # Precompute binary lifting table
    log = 60  # since K can be up to 10^18
    table = [array.array('I', [0]*N) for _ in range(log)]
    table[0] = array.array('I', X)
    
    for j in range(1, log):
        for i in range(N):
            table[j][i] = table[j-1][table[j-1][i]]
    
    # Compute f^K(i) for each i
    result = []
    for i in range(N):
        current = i
        for j in range(log):
            if K & (1 << j):
                current = table[j][current]
        result.append(A[current])
    
    # Print the result
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()