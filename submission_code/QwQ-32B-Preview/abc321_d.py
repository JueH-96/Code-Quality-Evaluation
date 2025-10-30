import sys
import bisect

def main():
    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    P = int(input[2])
    A = list(map(int, input[3:3+N]))
    B = list(map(int, input[3+N:3+N+M]))
    
    # Sort B
    B.sort()
    
    # Compute prefix sums of B
    prefix_sum = [0] * (M + 1)
    for i in range(M):
        prefix_sum[i+1] = prefix_sum[i] + B[i]
    
    # Calculate sum(A) and sum(B)
    sum_A = sum(A)
    sum_B = prefix_sum[M]
    
    # Initialize S'
    S_prime = 0
    
    for a in A:
        # Find the smallest j such that B[j] >= P - a
        x = P - a
        j = bisect.bisect_left(B, x)
        cnt = M - j
        if cnt == 0:
            continue
        sum_B_j = prefix_sum[M] - prefix_sum[j]
        sum_A_i = a * cnt
        sum_P = P * cnt
        S_prime += sum_A_i + sum_B_j - sum_P
    
    # Total sum S = sum(A)*M + sum(B)*N - S'
    total_sum = sum_A * M + sum_B * N - S_prime
    print(total_sum)

if __name__ == "__main__":
    main()