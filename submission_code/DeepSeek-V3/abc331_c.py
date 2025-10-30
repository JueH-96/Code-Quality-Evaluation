# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Create a sorted list of unique elements
    sorted_A = sorted(A)
    
    # Precompute the prefix sum of the sorted list
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + sorted_A[i]
    
    # For each element in A, find the sum of elements greater than it
    result = []
    for a in A:
        # Find the index where a would be inserted to keep the list sorted
        idx = bisect.bisect_right(sorted_A, a)
        # The sum of elements greater than a is the total sum minus the sum up to idx
        total_sum = prefix_sum[N]
        sum_up_to_idx = prefix_sum[idx]
        sum_greater = total_sum - sum_up_to_idx
        result.append(str(sum_greater))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()