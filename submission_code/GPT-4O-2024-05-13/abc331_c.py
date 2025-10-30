# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Create a sorted version of A
    sorted_A = sorted(A)
    
    # Create a prefix sum array for sorted_A
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + sorted_A[i]
    
    result = []
    for i in range(N):
        # Find the first element in sorted_A that is greater than A[i]
        idx = bisect.bisect_right(sorted_A, A[i])
        # Sum of all elements greater than A[i]
        sum_greater = prefix_sum[N] - prefix_sum[idx]
        result.append(sum_greater)
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()