import sys
import threading

def main():
    import sys
    data = sys.stdin
    N_M = data.readline().split()
    if not N_M:
        return
    N = int(N_M[0]); M = int(N_M[1])
    parents = data.readline().split()
    # build tree
    Np = N + 1
    children = [[] for _ in range(Np)]
    for i, p in enumerate(parents, start=2):
        pi = int(p)
        children[pi].append(i)
    # read events
    events_at_node = [[] for _ in range(Np)]
    for _ in range(M):
        line = data.readline().split()
        x = int(line[0]); y = int(line[1])
        events_at_node[x].append(y)
    # prepare for DFS
    depth = [0] * (Np)
    rem_list = [[] for _ in range(Np)]
    import heapq
    add_heap = []     # will store -end_depth
    remove_heap = []  # lazy removals, store same
    covered = 0
    # iterative DFS stack: (node, is_exit_flag)
    stack = [(1, False)]
    depth[1] = 0
    while stack:
        node, is_exit = stack.pop()
        if not is_exit:
            # entering node
            # add all events that start here
            for y in events_at_node[node]:
                endd = depth[node] + y
                rem_list[node].append(endd)
                heapq.heappush(add_heap, -endd)
            # clean up lazy removals
            while add_heap and remove_heap and add_heap[0] == remove_heap[0]:
                heapq.heappop(add_heap)
                heapq.heappop(remove_heap)
            # query coverage for this node
            if add_heap:
                # top of add_heap is negative of max end_depth
                if -add_heap[0] >= depth[node]:
                    covered += 1
            # push exit marker
            stack.append((node, True))
            # push children for DFS
            # reverse to preserve original order (not strictly needed)
            ch = children[node]
            for c in reversed(ch):
                depth[c] = depth[node] + 1
                stack.append((c, False))
        else:
            # exiting node: remove its events
            for endd in rem_list[node]:
                heapq.heappush(remove_heap, -endd)
            # clean up lazy removals again
            while add_heap and remove_heap and add_heap[0] == remove_heap[0]:
                heapq.heappop(add_heap)
                heapq.heappop(remove_heap)
    # output result
    sys.stdout.write(str(covered))

if __name__ == "__main__":
    main()