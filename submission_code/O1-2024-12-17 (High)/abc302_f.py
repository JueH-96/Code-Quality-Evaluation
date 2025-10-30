def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    # parse inputs
    N = int(input_data[0])
    M = int(input_data[1])
    
    idx = 2
    sets_list = []
    has_one = [False]*N
    has_m   = [False]*N
    
    # read each set
    for i in range(N):
        A_i = int(input_data[idx]); idx += 1
        s_i = list(map(int, input_data[idx:idx+A_i]))
        idx += A_i
        sets_list.append(s_i)
        # check if this set contains 1 or M
        if 1 in s_i:
            has_one[i] = True
        if M in s_i:
            has_m[i] = True
        # if a set already contains both 1 and M, answer is 0
        if has_one[i] and has_m[i]:
            print(0)
            return
    
    # Build "inverse" list: for each element e, which sets contain e?
    # We'll store 0-based set indices in inv[e].
    inv = [[] for _ in range(M+1)]
    for i, s_i in enumerate(sets_list):
        for e in s_i:
            inv[e].append(i)

    # We'll do a BFS over sets (0..N-1),
    # starting from all sets that contain 1.
    from collections import deque
    dist = [-1]*N
    visited = [False]*N
    used_element = [False]*(M+1)

    queue = deque()
    # enqueue all sets containing 1 (distance 0)
    for i in range(N):
        if has_one[i]:
            dist[i] = 0
            visited[i] = True
            queue.append(i)
    
    # if no set contains 1, we cannot succeed
    if not queue:
        print(-1)
        return
    
    # BFS
    while queue:
        cur = queue.popleft()
        # if current set contains M, return distance
        if has_m[cur]:
            print(dist[cur])
            return
        # expand neighbors (all sets that share an element)
        for e in sets_list[cur]:
            if not used_element[e]:
                used_element[e] = True
                for nxt in inv[e]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
    
    # if we exit BFS without finding a set containing M, it's impossible
    print(-1)

# Don't forget to call main()
if __name__ == "__main__":
    main()