import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    adj = [[] for _ in range(N + 1)]  # 1-based
    sum_c = 0
    for _ in range(N - 1):
        A = int(data[idx])
        B = int(data[idx + 1])
        C = int(data[idx + 2])
        idx += 3
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_c += C
    
    def find_farthest(start):
        max_dist = 0
        farthest_node = start
        stack = [(start, -1, 0)]
        while stack:
            node, parent, dist = stack.pop()
            if dist > max_dist:
                max_dist = dist
                farthest_node = node
            for neighbor, cost in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, dist + cost))
        return (farthest_node, max_dist)
    
    u, _ = find_farthest(1)
    v, diameter = find_farthest(u)
    ans = 2 * sum_c - diameter
    print(ans)

if __name__ == '__main__':
    main()