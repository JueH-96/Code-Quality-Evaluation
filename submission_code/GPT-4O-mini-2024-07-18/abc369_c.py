def count_arithmetic_progressions(N, A):
    count = 0
    
    # Each single element is an arithmetic progression
    count += N
    
    # To find pairs (l, r) where A[l:r] is an arithmetic progression
    for i in range(N):
        # Start a new subsequence from A[i]
        if i + 1 < N:
            d = A[i + 1] - A[i]
            length = 2  # We already have A[i] and A[i + 1]
            for j in range(i + 2, N):
                if A[j] - A[j - 1] == d:
                    length += 1
                else:
                    break
                count += 1  # Each extension of the subsequence is valid
            # Move the starting point to the next element
            i += length - 1  # Skip to the end of the current arithmetic progression
    
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N + 1]))

result = count_arithmetic_progressions(N, A)
print(result)