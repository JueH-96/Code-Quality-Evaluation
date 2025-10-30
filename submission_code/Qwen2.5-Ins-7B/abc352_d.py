# N K
# P = list(map(int, input().split()))
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Create a dictionary to store the index of each number in P
index_map = {P[i]: i for i in range(N)}

# Initialize the answer with a large number
min_diff = N

# Iterate through each number in the range [min(P), max(P) - K + 1]
for num in range(1, N + 1):
    if num + K - 1 > N:
        break
    start = index_map[num]
    end = index_map[num + K - 1]
    min_diff = min(min_diff, end - start)

print(min_diff)