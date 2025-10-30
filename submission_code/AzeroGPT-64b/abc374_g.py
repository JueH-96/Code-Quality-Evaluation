from collections import defaultdict

N = int(input())
S = [input() for _ in range(N)]

S_set = set(S)

    
graph = defaultdict(set)

for s in S:
    graph[s[0]].add(s[1])
    
for v in graph.values():
    v -= {None}
    
visited = set()

# stack implementation using list
def dfs(start):
    stack = [start]
    visited.add(start)
    while stack:
        cur = stack[-1]
        # print(cur)
        next = None
        # dfs
        for n in graph[cur] - visited:
            visited.add(n)
            stack.append(n)
            break
        else:
            stack.pop()
            # print(cur, "poped")
            if len(stack) >= 2:
                n = stack[-2]
                overlap_key = cur + n
                if overlap_key not in S_set:
                    return False
        
    return True

# 최대 나올 수 있는 DFS 리스프

cnt = 0
for start in graph:
    if start in visited:
        continue
    if not dfs(start):
        cnt += 1
        temp = []
        stack = [start]
        while stack:
            cur = stack.pop()
            temp.append(cur)
            for n in graph[cur]:
                if n not in visited:
                    visited.add(n)
                    stack.append(n)
        temp.append(start)
        # print(''.join(temp))

print(cnt)