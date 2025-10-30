# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Sort A and prepare prefix sums
    sorted_A = sorted(A)
    prefix_sum = [0] * (N + 1)
    
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + sorted_A[i - 1]
    
    total_sum = prefix_sum[N]
    
    # Function to find the first index in sorted_A where value is greater than x
    def find_first_greater(x):
        low, high = 0, N
        while low < high:
            mid = (low + high) // 2
            if sorted_A[mid] > x:
                high = mid
            else:
                low = mid + 1
        return low
    
    results = []
    for i in range(N):
        x = A[i]
        idx = find_first_greater(x)
        sum_greater = total_sum - prefix_sum[idx]
        results.append(sum_greater)
    
    print(" ".join(map(str, results)))