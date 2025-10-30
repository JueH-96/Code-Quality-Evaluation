def main():
    import sys
    def input():
        return sys.stdin.readline()
    
    # Read N and X
    N_X = input().strip()
    while N_X == '':
        N_X = input().strip()
    N, X = map(int, N_X.split())
    
    # Read array A
    A = []
    while len(A) < N:
        line = input()
        if not line:
            continue
        A += list(map(int, line.strip().split()))
    
    # Create list of (value, original_index)
    A_with_index = [(A[i], i+1) for i in range(N)]
    
    # Sort the list based on value
    A_with_index.sort(key=lambda x: x[0])
    
    # Iterate through all possible triples
    for i in range(N-2):
        left = i + 1
        right = N - 1
        while left < right:
            current_sum = A_with_index[i][0] + A_with_index[left][0] + A_with_index[right][0]
            if current_sum == X:
                # Check if original indices are in increasing order
                if A_with_index[i][1] < A_with_index[left][1] < A_with_index[right][1]:
                    print(A_with_index[i][1], A_with_index[left][1], A_with_index[right][1])
                    return
                else:
                    # Move pointers to find another combination
                    left += 1
                    right -= 1
            elif current_sum < X:
                left += 1
            else:
                right -= 1
    
    # If no triple found
    print(-1)

if __name__ == '__main__':
    main()