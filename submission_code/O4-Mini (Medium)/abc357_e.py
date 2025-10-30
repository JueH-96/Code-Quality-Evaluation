import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:]))

    # convert to 0-based
    to = [x-1 for x in a]

    visited = [False]*n
    index_in_path = [-1]*n
    cycle_size = [0]*n
    dist = [0]*n

    for u in range(n):
        if visited[u]:
            continue
        path = []
        curr = u
        # walk until we either hit a node in the current path (cycle)
        # or hit a node already fully processed
        while index_in_path[curr] == -1 and not visited[curr]:
            index_in_path[curr] = len(path)
            path.append(curr)
            curr = to[curr]

        if index_in_path[curr] != -1:
            # found a cycle in path
            start = index_in_path[curr]
            cyc_len = len(path) - start
            # assign cycle nodes
            for i in range(start, len(path)):
                w = path[i]
                cycle_size[w] = cyc_len
                dist[w] = 0
            # assign tail nodes leading into that cycle
            for i in range(start-1, -1, -1):
                w = path[i]
                wn = to[w]
                cycle_size[w] = cycle_size[wn]
                dist[w] = dist[wn] + 1
        else:
            # curr is already processed, just link our path into it
            for i in range(len(path)-1, -1, -1):
                w = path[i]
                wn = to[w]
                cycle_size[w] = cycle_size[wn]
                dist[w] = dist[wn] + 1

        # mark all in path as done, clear their index_in_path
        for w in path:
            visited[w] = True
            index_in_path[w] = -1

    # sum reachable counts: dist[u] + cycle_size[u]
    total = 0
    for i in range(n):
        total += dist[i] + cycle_size[i]

    print(total)

if __name__ == "__main__":
    main()