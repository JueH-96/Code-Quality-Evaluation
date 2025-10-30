import sys

def main():
    input = sys.stdin.readline
    data = input().split()
    if not data:
        return
    n = int(data[0])
    m = int(data[1])

    # build adjacency list
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    target = n
    # ans[0] will hold the minimum XOR found
    ans = [1 << 61]

    def dfs(u, visited_mask, xor_val):
        # if we've reached the target, update answer
        if u == target:
            if xor_val < ans[0]:
                ans[0] = xor_val
            return
        # otherwise, continue exploring
        for v, w in graph[u]:
            bit = 1 << (v - 1)
            if not (visited_mask & bit):
                dfs(v, visited_mask | bit, xor_val ^ w)

    # start DFS from node 1, mark it visited, initial xor = 0
    dfs(1, 1 << (1 - 1), 0)

    # print the result
    sys.stdout.write(str(ans[0]))

# call main to execute
main()