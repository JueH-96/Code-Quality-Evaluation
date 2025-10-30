def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Edge case: If N == 1, no operations are needed.
    if N == 1:
        print(0)
        return
    
    # Calculate the sum of all elements.
    total_sum = sum(A)
    
    # L is the floor average; R is how many elements must be L+1.
    L = total_sum // N
    R = total_sum % N
    
    # Sort the array
    A.sort()
    
    # Compute how many units must be moved ("surplus") from
    # elements exceeding their target value.
    # - The first (N - R) elements will aim for L.
    # - The last R elements will aim for L+1.
    cost = 0
    for i in range(N - R):
        if A[i] > L:
            cost += A[i] - L
    for i in range(N - R, N):
        if A[i] > L + 1:
            cost += A[i] - (L + 1)
    
    print(cost)

# Do not forget to call main()
main()