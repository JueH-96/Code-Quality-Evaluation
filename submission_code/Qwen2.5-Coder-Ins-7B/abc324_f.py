import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = []
for i in range(M):
    u = int(data[2 + i * 4]) - 1
    v = int(data[3 + i * 4]) - 1
    b = int(data[4 + i * 4])
    c = int(data[5 + i * 4])
    edges.append((u, v, b, c))

# Initialize the maximum beauty-to-cost ratio
max_ratio = 0

# Iterate through all possible paths from vertex 1 to vertex N
for i in range(M):
    for j in range(i + 1, M):
        for k in range(j + 1, M):
            u1, v1, b1, c1 = edges[i]
            u2, v2, b2, c2 = edges[j]
            u3, v3, b3, c3 = edges[k]
            
            # Check if the edges form a valid path from vertex 1 to vertex N
            if v1 == u2 and v2 == u3 and v3 == N - 1:
                total_beauty = b1 + b2 + b3
                total_cost = c1 + c2 + c3
                ratio = total_beauty / total_cost
                if ratio > max_ratio:
                    max_ratio = ratio

# Print the maximum beauty-to-cost ratio
print(max_ratio)