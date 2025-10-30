# YOUR CODE HERE
def solve():
    n, x, k = map(int, input().split())
    
    count = 0
    q = [(x, 0)]
    visited = {x}
    
    while q:
        curr, dist = q.pop(0)
        if dist == k:
            count += 1
        elif dist < k:
            next_nodes = []
            if 2 * curr <= n:
                next_nodes.append(2 * curr)
            if curr -1 >=1 and curr %2 == 0:
                next_nodes.append(curr -1)
            if curr -1 >=1 and curr %2 != 0:
                next_nodes.append(curr -1)
            
            for next_node in next_nodes:
                if next_node not in visited:
                    visited.add(next_node)
                    q.append((next_node, dist + 1))
    print(count)

t = int(input())
for _ in range(t):
    solve()