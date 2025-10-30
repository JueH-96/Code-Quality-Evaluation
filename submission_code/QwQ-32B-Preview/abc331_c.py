import sys
import bisect

def main():
    # Read all input
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Sort the array
    A_sorted = sorted(A)
    
    # Compute prefix sums
    prefix_sum = [0] * (N + 1)
    for k in range(1, N + 1):
        prefix_sum[k] = prefix_sum[k-1] + A_sorted[k-1]
    
    # Calculate sums for each A_i
    result = []
    for a in A:
        k = bisect.bisect_right(A_sorted, a)
        if k < N:
            sum_greater = prefix_sum[N] - prefix_sum[k]
        else:
            sum_greater = 0
        result.append(str(sum_greater))
    
    # Print the result
    print(' '.join(result))

if __name__ == "__main__":
    main()