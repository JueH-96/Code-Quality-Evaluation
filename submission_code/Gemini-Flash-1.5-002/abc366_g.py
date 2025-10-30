# YOUR CODE HERE
n, m = map(int, input().split())
edges = []
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    u -= 1
    v -= 1
    edges.append((u, v))
    graph[u].append(v)
    graph[v].append(u)

if n == 1 and m == 0:
    print("Yes")
    print(1)
elif m == 0:
    print("Yes")
    print(*([1] * n))
else:
    ans = [0] * n
    found = False
    for i in range(1 << n):
        temp_ans = [0] * n
        for j in range(n):
            if (i >> j) & 1:
                temp_ans[j] = 1
            else:
                temp_ans[j] = 0

        valid = True
        for j in range(n):
            if len(graph[j]) > 0:
                xor_sum = 0
                for neighbor in graph[j]:
                    xor_sum ^= temp_ans[neighbor]
                if xor_sum != 0:
                    valid = False
                    break
        if valid:
            found = True
            ans = temp_ans
            break
    if found:
        print("Yes")
        print(*ans)
    else:
        print("No")