def count_kagamimochi(N, A):
    count = 0
    j = 0
    
    for i in range(N):
        while j < N and A[j] <= 2 * A[i]:
            j += 1
        count += j - i - 1  # j is now one past the valid range, subtract 1 for the current mochi
    
    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Calculate and print the result
result = count_kagamimochi(N, A)
print(result)