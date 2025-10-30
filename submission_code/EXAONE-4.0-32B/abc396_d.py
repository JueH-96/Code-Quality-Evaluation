import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n+1)]
    
    idx = 2
    for _ in range(m):
        u = int(data[idx])
        v = int(data[idx+1])
        w = int(data[idx+2])
        idx += 3
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    stack = [(1, 1, 0)]
    best = None
    
    while stack:
        node, mask, xor_val = stack.pop()
        if node == n:
            if best is None or xor_val < best:
                best = xor_val
            continue
        
        for neighbor, weight in graph[node]:
            bit = 1 << (neighbor - 1)
            if mask & bit:
                continue
            new_mask = mask | bit
            new_xor = xor_val ^ weight
            stack.append((neighbor, new_mask, new_xor))
    
    print(best)

if __name__ == "__main__":
    main()