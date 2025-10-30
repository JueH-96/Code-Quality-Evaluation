import sys
import bisect

def main():
    import sys
    from bisect import bisect_right
    from itertools import accumulate

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    K = int(data[2*N+1])
    queries = list(zip(map(int, data[2*N+2:2*N+2+2*K:2]), map(int, data[2*N+3:2*N+2+2*K:2])))

    # Sort A and B
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    
    # Compute prefix sums for A and B
    prefix_A = list(accumulate(A_sorted, initial=0))
    prefix_B = list(accumulate(B_sorted, initial=0))

    for X_k, Y_k in queries:
        X_k -= 1  # Adjust to 0-based indexing
        Y_k -= 1  # Adjust to 0-based indexing
        A_prime = A_sorted[:X_k+1]
        B_prime = B_sorted[:Y_k+1]
        
        sum_B = prefix_B[Y_k+1]
        total_sum = 0
        P = 0
        sum_B_less = 0
        for a in A_prime:
            while P <= Y_k and B_prime[P] <= a:
                sum_B_less += B_prime[P]
                P += 1
            term1 = a * (2 * P - (Y_k + 1))
            term2 = sum_B - 2 * sum_B_less
            total_sum += term1 + term2
        print(total_sum)

if __name__ == "__main__":
    main()