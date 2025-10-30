import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    n = int(data[0])
    edges_set = set()
    graph = {}
    
    for i in range(1, 1+n):
        s = data[i].strip()
        if len(s) < 2:
            continue
        u = s[0]
        v = s[1]
        edge = (u, v)
        edges_set.add(edge)
        if u not in graph:
            graph[u] = []
        graph[u].append(v)
        if v not in graph:
            graph[v] = []
            
    covered = set()
    walks = 0
    
    while edges_set - covered:
        candidates = []
        for u in graph:
            count = 0
            if u in graph:
                for v in graph[u]:
                    edge = (u, v)
                    if edge in edges_set and edge not in covered:
                        count += 1
            if count > 0:
                candidates.append((u, count))
                
        if not candidates:
            break
            
        candidates.sort(key=lambda x: (-x[1], x[0]))
        start_vertex = candidates[0][0]
        current = start_vertex
        
        while True:
            found_uncovered = False
            next_node = None
            max_count = -1
            if current in graph:
                for v in graph[current]:
                    edge = (current, v)
                    if edge in edges_set and edge not in covered:
                        found_uncovered = True
                        count_v = 0
                        if v in graph:
                            for w in graph[v]:
                                e = (v, w)
                                if e in edges_set and e not in covered:
                                    count_v += 1
                        if count_v > max_count:
                            max_count = count_v
                            next_node = v
            if found_uncovered:
                edge_to_cover = (current, next_node)
                covered.add(edge_to_cover)
                current = next_node
            else:
                parent_map = {}
                visited = set()
                queue = deque()
                queue.append(current)
                visited.add(current)
                target_vertex = None
                while queue:
                    u = queue.popleft()
                    if u in graph:
                        for v in graph[u]:
                            edge_uv = (u, v)
                            if edge_uv in edges_set and edge_uv not in covered:
                                target_vertex = v
                                parent_map[v] = u
                                break
                    if target_vertex is not None:
                        break
                    if u in graph:
                        for v in graph[u]:
                            if v not in visited:
                                visited.add(v)
                                parent_map[v] = u
                                queue.append(v)
                if target_vertex is None:
                    break
                else:
                    path_edges = []
                    node_chain = []
                    temp = target_vertex
                    while temp != current:
                        node_chain.append(temp)
                        temp = parent_map[temp]
                    node_chain.reverse()
                    prev_node = current
                    for node in node_chain:
                        edge_here = (prev_node, node)
                        path_edges.append(edge_here)
                        prev_node = node
                    for edge in path_edges:
                        if edge in edges_set and edge not in covered:
                            covered.add(edge)
                    current = target_vertex
                    
        walks += 1
        
    print(walks)

if __name__ == "__main__":
    main()