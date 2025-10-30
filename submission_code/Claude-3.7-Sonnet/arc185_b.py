def is_possible_to_make_non_decreasing(A, N):
    # Compute an ideal non-decreasing array B with the same sum as A
    total_sum = sum(A)
    B = [0] * N
    distributed_sum = 0
    
    for i in range(N):
        if i == 0:
            B[i] = total_sum // N
        else:
            remaining = total_sum - distributed_sum
            count = N - i
            B[i] = max(B[i-1], remaining // count)
        
        distributed_sum += B[i]
    
    # Adjust the last element to make the sum of B equal to the sum of A
    B[N-1] += total_sum - distributed_sum
    
    # Ensure B is still non-decreasing after the adjustment
    if N > 1 and B[N-1] < B[N-2]:
        return "No"
    
    # Check if the prefix sums of A are at most the prefix sums of B
    A_prefix_sum = 0
    B_prefix_sum = 0
    
    for i in range(N):
        A_prefix_sum += A[i]
        B_prefix_sum += B[i]
        
        if A_prefix_sum > B_prefix_sum:
            return "No"
    
    return "Yes"

def main():
    T = int(input())
    
    for _ in range(T):
        N = int(input())
        A = list(map(int, input().split()))
        print(is_possible_to_make_non_decreasing(A, N))

if __name__ == "__main__":
    main()