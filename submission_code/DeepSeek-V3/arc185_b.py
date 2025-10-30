def can_make_non_decreasing(N, A):
    # Calculate the prefix sum of the original array
    prefix_sum = [0] * N
    prefix_sum[0] = A[0]
    for i in range(1, N):
        prefix_sum[i] = prefix_sum[i-1] + A[i]
    
    # Calculate the required prefix sum for a non-decreasing sequence
    # The required sequence is [A[0], A[0], ..., A[0]]
    # Because in the non-decreasing sequence, the first element is the smallest
    # and all elements are at least as large as the first.
    # So, the required prefix sum is [A[0], 2*A[0], 3*A[0], ..., N*A[0]]
    required_prefix_sum = [A[0] * (i+1) for i in range(N)]
    
    # Check if the original prefix sum is at least the required prefix sum
    for i in range(N):
        if prefix_sum[i] < required_prefix_sum[i]:
            return False
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        A = list(map(int, data[idx:idx+N]))
        idx += N
        if can_make_non_decreasing(N, A):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()