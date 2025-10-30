# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

graph = [[] for _ in range(n)]
for i in range(n):
    graph[i].append(a[i] - 1)

count = 0
for i in range(n):
    reachable = set()
    curr = i
    while curr not in reachable:
        reachable.add(curr)
        curr = graph[curr][0]
    for j in reachable:
        count += 1

print(count)