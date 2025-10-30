def longest_increasing_subsequence(A, R, X):
    # We will use a dynamic programming approach with binary search to find the LIS
    from bisect import bisect_left
    
    # This will store the length of the longest increasing subsequence found
    lis = []
    
    for i in range(R):
        if A[i] <= X:
            pos = bisect_left(lis, A[i])
            if pos == len(lis):
                lis.append(A[i])
            else:
                lis[pos] = A[i]
    
    return len(lis)

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and Q
    N, Q = map(int, data[0].split())
    
    # Read the sequence A
    A = list(map(int, data[1].split()))
    
    results = []
    
    # Process each query
    for i in range(2, 2 + Q):
        R, X = map(int, data[i].split())
        result = longest_increasing_subsequence(A[:R], R, X)
        results.append(result)
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()