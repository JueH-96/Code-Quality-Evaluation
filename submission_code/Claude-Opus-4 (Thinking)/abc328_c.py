# YOUR CODE HERE
N, Q = map(int, input().split())
S = input()

# Create prefix sum array
# prefix[i] = number of positions j < i where S[j] == S[j+1]
prefix = [0] * N
for i in range(1, N):
    prefix[i] = prefix[i-1]
    if S[i-1] == S[i]:
        prefix[i] += 1

# Process queries
for _ in range(Q):
    l, r = map(int, input().split())
    # Count positions p where l <= p <= r-1 and S[p] == S[p+1]
    # In 0-indexed: count positions where l-1 <= j <= r-2 and S[j] == S[j+1]
    answer = prefix[r-1] - prefix[l-1]
    print(answer)