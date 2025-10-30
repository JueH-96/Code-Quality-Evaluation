def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # To store the last occurrence index of each element
    last_occurrence = {}
    
    # To store the result
    result = 0
    
    # To store the current sum of f(i,j) for all j >= i
    current_sum = 0
    
    for i in range(N):
        if A[i] in last_occurrence:
            # Subtract the previous contribution of A[i]
            current_sum -= (last_occurrence[A[i]] + 1)
        # Add the current contribution of A[i]
        current_sum += (i + 1)
        # Update the last occurrence of A[i]
        last_occurrence[A[i]] = i
        # Add the current_sum to the result
        result += current_sum
    
    print(result)

if __name__ == "__main__":
    main()