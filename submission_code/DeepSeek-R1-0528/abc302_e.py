import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    q = int(data[1])
    index = 2
    deg = [0] * (n + 1)
    adj = [set() for _ in range(n + 1)]
    isolated_count = n
    out_lines = []
    
    for _ in range(q):
        t = data[index]
        index += 1
        if t == '1':
            u = int(data[index])
            v = int(data[index + 1])
            index += 2
            if deg[u] == 0:
                isolated_count -= 1
            if deg[v] == 0:
                isolated_count -= 1
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            out_lines.append(str(isolated_count))
        else:
            v_val = int(data[index])
            index += 1
            neighbors = list(adj[v_val])
            adj[v_val] = set()
            old_deg = deg[v_val]
            deg[v_val] = 0
            for w in neighbors:
                if v_val in adj[w]:
                    adj[w].discard(v_val)
                    deg[w] -= 1
                    if deg[w] == 0:
                        isolated_count += 1
            if old_deg > 0:
                isolated_count += 1
            out_lines.append(str(isolated_count))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()