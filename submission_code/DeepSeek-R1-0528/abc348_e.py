import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n)]
    index = 1
    for i in range(n-1):
        a = int(data[index]); b = int(data[index+1]); index += 2
        graph[a-1].append(b-1)
        graph[b-1].append(a-1)
    
    C = list(map(int, data[index:index+n]))
    index += n
    total_all = sum(C)
    
    parent_arr = [-1] * n
    depth = [0] * n
    children = [[] for _ in range(n)]
    stack = [0]
    parent_arr[0] = -1
    while stack:
        u = stack.pop()
        for v in graph[u]:
            if v == parent_arr[u]:
                continue
            parent_arr[v] = u
            children[u].append(v)
            depth[v] = depth[u] + 1
            stack.append(v)
    
    order = []
    stack = [0]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)
    
    total_weight = [0] * n
    for u in order[::-1]:
        total_weight[u] = C[u]
        for v in children[u]:
            total_weight[u] += total_weight[v]
    
    f0 = 0
    for i in range(n):
        f0 += C[i] * depth[i]
    
    f_vals = [0] * n
    f_vals[0] = f0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in children[u]:
            f_vals[v] = f_vals[u] + total_all - 2 * total_weight[v]
            stack.append(v)
    
    ans = min(f_vals)
    print(ans)

if __name__ == '__main__':
    main()