# YOUR CODE HERE
n = int(input())
edges = []
for i in range(n // 2):
    a, b = map(int, input().split())
    edges.append((a, b))
    
operations = []
for i in range(n // 2):
    a, b = edges[i]
    operations.append((a, b))
    
for i in range(n // 2, n - 1):
    a, b = edges[i]
    operations.append((a, b))
    
for i in range(n // 2):
    print(operations[i][0], operations[i][1])