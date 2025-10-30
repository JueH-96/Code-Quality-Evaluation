import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u = int(input[ptr])
        v = int(input[ptr + 1])
        adj[u].append(v)
        adj[v].append(u)
        ptr += 2
    
    max_size = 0
    
    for u in range(1, N + 1):
        neighbors = adj[u]
        m = len(neighbors)
        if m == 0:
            continue
        list_a = [len(adj[v]) - 1 for v in neighbors]
        list_a.sort(reverse=True)
        
        current_max = 0
        for x in range(1, m + 1):
            val = 1 + x + x * list_a[x-1]
            if val > current_max:
                current_max = val
        if current_max > max_size:
            max_size = current_max
    
    print(N - max_size)

if __name__ == "__main__":
    main()