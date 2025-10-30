import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    parents = [0] * (n + 1)
    p_list = list(map(int, sys.stdin.readline().split()))
    for i in range(2, n + 1):
        parents[i] = p_list[i - 2]
    
    max_y = [0] * (n + 1)
    for _ in range(m):
        x, y = map(int, sys.stdin.readline().split())
        if y > max_y[x]:
            max_y[x] = y
    
    depth = [0] * (n + 1)
    for i in range(2, n + 1):
        depth[i] = depth[parents[i]] + 1
    
    max_s = [0] * (n + 1)
    for u in range(1, n + 1):
        if max_y[u] > 0:
            current_s = max_y[u] + depth[u]
            if u == 1:
                max_s[u] = current_s
            else:
                parent = parents[u]
                max_s[u] = max(current_s, max_s[parent])
        else:
            if u == 1:
                max_s[u] = -1  # Indicates no valid coverage
            else:
                parent = parents[u]
                max_s[u] = max_s[parent]
    
    count = 0
    for u in range(1, n + 1):
        if max_s[u] >= depth[u]:
            count += 1
    print(count)

if __name__ == "__main__":
    main()