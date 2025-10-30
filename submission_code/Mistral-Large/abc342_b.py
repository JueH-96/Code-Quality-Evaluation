import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = []

index = N + 2
for _ in range(Q):
    A = int(data[index])
    B = int(data[index + 1])
    queries.append((A, B))
    index += 2

# Create a mapping from person number to their position
position_map = {person: pos for pos, person in enumerate(P)}

results = []
for A, B in queries:
    pos_A = position_map[A]
    pos_B = position_map[B]
    if pos_A < pos_B:
        results.append(A)
    else:
        results.append(B)

# Print results
for result in results:
    print(result)