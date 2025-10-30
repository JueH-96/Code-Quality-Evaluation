def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # 1) Compute the total sum S of all (A_i + A_j) for i < j (without the mod).
    #    We'll do this in O(N) using a prefix sum approach.
    prefix_sum = [0]*(N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + A[i]
    
    S = 0
    # sum_{0 <= i < j < N} (A[j] + A[i]) = sum_{j=1..N-1} [ A[j]*j + sum_{i=0..j-1} A[i] ]
    #                                   = sum_{j=1..N-1} [ A[j]*j + prefix_sum[j] ]
    for j in range(1, N):
        S += A[j]*j + prefix_sum[j]
    
    # 2) Sort the array to efficiently count the number of pairs (i < j) with A[i] + A[j] >= 10^8.
    A.sort()
    
    # 3) Two-pointer approach to count M = number of pairs with sum >= 10^8
    M = 0
    left, right = 0, N-1
    TH = 10**8
    while left < right:
        if A[left] + A[right] >= TH:
            M += (right - left)
            right -= 1
        else:
            left += 1
    
    # 4) Each such pair contributes a subtraction of 10^8 (since f(x,y) = (x+y) % 10^8).
    answer = S - M*(10**8)
    
    # Print the final result
    print(answer)

# Do not forget to call main!
if __name__ == "__main__":
    main()