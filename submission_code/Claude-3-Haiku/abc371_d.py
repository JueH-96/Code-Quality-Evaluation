# Read the input
N = int(input())
X = list(map(int, input().split()))
P = list(map(int, input().split()))
Q = int(input())
queries = []
for _ in range(Q):
    L, R = map(int, input().split())
    queries.append((L, R))

# Solve the problem
def count_villagers(L, R):
    total = 0
    for i in range(N):
        if L <= X[i] <= R:
            total += P[i]
    return total

for L, R in queries:
    print(count_villagers(L, R))