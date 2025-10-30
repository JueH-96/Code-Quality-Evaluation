import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    a = [i-1 for i in a]
    visited = [0]*N
    ans = [0]*N
    for i in range(N):
        if visited[i] == 0:
            stack = deque([i])
            path = []
            while stack:
                v = stack[-1]
                if visited[v] == 0:
                    visited[v] = 1
                    path.append(v)
                    stack.append(a[v])
                elif visited[v] == 1:
                    loop = path[path.index(a[v]):]
                    loop_len = len(loop)
                    for v in loop:
                        ans[v] = loop_len
                    path = path[:path.index(a[v])]
                    stack.pop()
                else:
                    stack.pop()
            path_len = len(path)
            for v in path:
                ans[v] = path_len + ans[a[v]]
    print(sum(ans))

if __name__ == "__main__":
    main()