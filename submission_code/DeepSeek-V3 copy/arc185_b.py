def can_make_non_decreasing(N, A):
    # Calculate the prefix sum of the original array
    prefix_original = [0] * N
    prefix_original[0] = A[0]
    for i in range(1, N):
        prefix_original[i] = prefix_original[i-1] + A[i]
    
    # Sort the array to get the target non-decreasing sequence
    sorted_A = sorted(A)
    
    # Calculate the prefix sum of the sorted array
    prefix_sorted = [0] * N
    prefix_sorted[0] = sorted_A[0]
    for i in range(1, N):
        prefix_sorted[i] = prefix_sorted[i-1] + sorted_A[i]
    
    # Check if the prefix sums match
    for i in range(N):
        if prefix_original[i] < prefix_sorted[i]:
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