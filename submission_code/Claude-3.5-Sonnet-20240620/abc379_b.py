# YOUR CODE HERE
def max_strawberries(N, K, S):
    count = 0
    i = 0
    while i < N:
        if S[i:i+K] == 'O' * K:
            count += 1
            S = S[:i] + 'X' * K + S[i+K:]
            i += K
        else:
            i += 1
    return count

# Read input
N, K = map(int, input().split())
S = input().strip()

# Calculate and print the result
result = max_strawberries(N, K, S)
print(result)