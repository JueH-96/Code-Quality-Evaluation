# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    possible_a = set()
    for i in range(n):
        reachable = set()
        q = [i]
        visited = [False] * n
        visited[i] = True
        reachable.add(a[i])
        
        while q:
            curr = q.pop(0)
            for next_node in range(n):
                if abs(curr - next_node) <= k and not visited[next_node]:
                    visited[next_node] = True
                    reachable.add(a[next_node])
                    q.append(next_node)
        possible_a.add(tuple(sorted(list(reachable))))
    
    for i in range(n):
        found = False
        for reachable_set in possible_a:
            if b[i] in reachable_set:
                found = True
                break
        if not found:
            print("No")
            return
    
    print("Yes")


t = int(input())
for _ in range(t):
    solve()