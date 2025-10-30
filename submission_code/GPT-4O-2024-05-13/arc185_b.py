# YOUR CODE HERE
import sys
input = sys.stdin.read

def can_be_non_decreasing(N, A):
    # We need to check if we can make the array non-decreasing
    # by using the allowed operations.
    # The key observation is that we can always move excess values
    # from left to right to make the array non-decreasing.
    
    # We will use a greedy approach to check if we can make the array
    # non-decreasing by ensuring that each element is at least as large
    # as the previous one.
    
    for i in range(1, N):
        if A[i] < A[i - 1]:
            # If A[i] is less than A[i-1], we need to check if we can
            # move enough values from A[i-1] to A[i] to make A[i] >= A[i-1].
            # This is possible if the sum of the elements up to i is
            # non-decreasing.
            if A[i] + i < A[i - 1]:
                return "No"
            A[i] = max(A[i], A[i - 1])
    
    return "Yes"

def main():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = list(map(int, data[index:index + N]))
        index += N
        results.append(can_be_non_decreasing(N, A))
    
    print("
".join(results))

if __name__ == "__main__":
    main()