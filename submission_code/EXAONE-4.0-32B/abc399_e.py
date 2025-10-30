import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return
    n = int(data[0])
    S = data[1].strip()
    T = data[2].strip()
    
    mapping = {}
    for i in range(n):
        s_char = S[i]
        t_char = T[i]
        if s_char in mapping:
            if mapping[s_char] != t_char:
                print(-1)
                return
        else:
            mapping[s_char] = t_char
    
    graph = {}
    for char in mapping:
        if mapping[char] != char:
            graph[char] = mapping[char]
    
    edges = len(graph)
    if edges == 0:
        print(0)
        return
    
    visited = set()
    cycles = 0
    for node in list(graph.keys()):
        if node not in visited:
            path = []
            cur = node
            while cur is not None and cur not in visited:
                visited.add(cur)
                path.append(cur)
                if cur in graph:
                    cur = graph[cur]
                else:
                    cur = None
            if cur is not None and cur in path:
                cycles += 1
                
    total_ops = edges + cycles
    print(total_ops)

if __name__ == "__main__":
    main()