def max_sum_after_operations(N, A):
    S = []
    for value in A:
        if value > 0:
            S.append(value)
        elif S:
            # Only pop if S is not empty
            S.pop()
    return sum(S)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Calculate and print the result
result = max_sum_after_operations(N, A)
print(result)