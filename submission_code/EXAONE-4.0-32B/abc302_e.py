import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    index = 2
    
    deg = [0] * (n + 1)
    graph = [set() for _ in range(n + 1)]
    isolated = n
    
    out_lines = []
    
    for _ in range(q):
        t = int(data[index])
        index += 1
        if t == 1:
            u = int(data[index])
            v = int(data[index + 1])
            index += 2
            if deg[u] == 0:
                isolated -= 1
            if deg[v] == 0:
                isolated -= 1
            graph[u].add(v)
            graph[v].add(u)
            deg[u] += 1
            deg[v] += 1
            out_lines.append(str(isolated))
        else:
            v = int(data[index])
            index += 1
            if deg[v] > 0:
                neighbors = list(graph[v])
                for w in neighbors:
                    if w == v:
                        continue
                    if v in graph[w]:
                        graph[w].remove(v)
                        deg[w] -= 1
                        if deg[w] == 0:
                            isolated += 1
                graph[v] = set()
                deg[v] = 0
                isolated += 1
            out_lines.append(str(isolated))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()