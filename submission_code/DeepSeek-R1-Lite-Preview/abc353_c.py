def main():
    MOD = 10**8
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Compute sum1 and sum2
    sum1 = sum(A[i] * (N - i - 1) for i in range(N-1))
    sum2 = sum(A[j] * j for j in range(1, N))
    sum_pairs = sum1 + sum2
    
    # Sort A in descending order
    A_sorted = sorted(A, reverse=True)
    
    # Count pairs where A[i] + A[j] >= MOD using two pointers
    left = 0
    right = N - 1
    count = 0
    while left < right:
        if A_sorted[left] + A_sorted[right] >= MOD:
            count += right - left
            left += 1
        else:
            right -= 1
    
    # Compute the final answer
    answer = sum_pairs - MOD * count
    print(answer)

if __name__ == '__main__':
    main()