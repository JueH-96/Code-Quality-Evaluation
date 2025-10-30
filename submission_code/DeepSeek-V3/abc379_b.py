# YOUR CODE HERE
def max_strawberries(N, K, S):
    count = 0
    S = list(S)
    i = 0
    while i <= N - K:
        if all(c == 'O' for c in S[i:i+K]):
            count += 1
            for j in range(i, i+K):
                S[j] = 'X'
            i += K
        else:
            i += 1
    return count

# Read input
N, K = map(int, input().split())
S = input().strip()

# Compute and print the result
print(max_strawberries(N, K, S))