def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    T = data[2].strip()
    # need[x] = y means letter x must end up as y
    need = [-1] * 26
    presentS = [False] * 26
    for ch in S:
        presentS[ord(ch) - 97] = True
    # build need mapping, check consistency
    for i in range(N):
        x = ord(S[i]) - 97
        y = ord(T[i]) - 97
        if need[x] == -1:
            need[x] = y
        elif need[x] != y:
            print(-1)
            return
    # count edges and prepare for cycle detection
    edges = 0
    for x in range(26):
        if need[x] != -1 and need[x] != x:
            edges += 1
    # colors: 0 unvisited, 1 visiting, 2 done
    color = [0] * 26
    def dfs(u):
        color[u] = 1
        v = need[u]
        # if there's a real outgoing edge
        if v != -1 and v != u:
            if color[v] == 0:
                if dfs(v):
                    color[u] = 2
                    return True
            elif color[v] == 1:
                # back-edge => found a cycle
                color[u] = 2
                return True
        color[u] = 2
        return False

    # count number of distinct cycles in the graph
    cycle_count = 0
    for x in range(26):
        if color[x] == 0:
            if dfs(x):
                cycle_count += 1

    # count free letters (letters not in initial S)
    free_count = sum(1 for x in range(26) if not presentS[x])
    # if there's a cycle but no free letter to break it, impossible
    if cycle_count > 0 and free_count == 0:
        print(-1)
    else:
        # minimum ops = edges + number of cycles
        print(edges + cycle_count)

if __name__ == "__main__":
    main()