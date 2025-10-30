def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    # Sort A and B in decreasing order
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)
    
    # Compute prefix sums
    prefix_A = [0] * (N + 1)
    prefix_B = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_A[i] = prefix_A[i-1] + A_sorted[i-1]
        prefix_B[i] = prefix_B[i-1] + B_sorted[i-1]
    
    # Binary search for the smallest k
    left = 1
    right = N
    answer = N
    while left <= right:
        mid = (left + right) // 2
        if prefix_A[mid] > X or prefix_B[mid] > Y:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(answer)

if __name__ == "__main__":
    main()