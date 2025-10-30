def main():
    import sys
    import bisect

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)
    
    prefix_A = [0]*(N+1)
    prefix_B = [0]*(N+1)
    
    for i in range(N):
        prefix_A[i+1] = prefix_A[i] + A_sorted[i]
        prefix_B[i+1] = prefix_B[i] + B_sorted[i]
    
    left = 1
    right = N
    answer = N
    
    while left <= right:
        mid = (left + right) // 2
        sum_A = prefix_A[mid]
        sum_B = prefix_B[mid]
        if sum_A > X or sum_B > Y:
            answer = mid
            right = mid -1
        else:
            left = mid +1
    
    # Check if no such k exists
    if prefix_A[N] <= X and prefix_B[N] <= Y:
        print(N)
    else:
        print(answer)

if __name__ == "__main__":
    main()