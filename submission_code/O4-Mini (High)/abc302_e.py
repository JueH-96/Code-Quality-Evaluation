import sys
from array import array

def main():
    input = sys.stdin.readline
    # Read N and Q
    N, Q = map(int, input().split())
    # Maximum number of neighbor‐list entries = 2 per add query
    max_entries = 2 * Q + 5

    # head[v] = index of first entry in v's adjacency list (0 means empty)
    head = array('I', [0]) * (N + 1)
    # For each entry idx:
    #   nb[idx]   = the neighbor vertex
    #   tadd[idx] = the "add‐serial" when this edge was added
    #   nxt[idx]  = next entry in the same adjacency list
    nb   = array('I', [0]) * max_entries
    tadd = array('I', [0]) * max_entries
    nxt  = array('I', [0]) * max_entries

    # deg[v] = current degree of v
    deg = array('I', [0]) * (N + 1)
    # clear_time[v] = the max "add‐serial" seen when v was last cleared
    clear_time = array('I', [0]) * (N + 1)

    isolated = N      # number of vertices with degree == 0
    add_count = 0     # serial ID for add‐operations
    ptr = 1           # next free index in neighbor‐entry arrays
    out = []

    for _ in range(Q):
        qry = input().split()
        if qry[0] == '1':
            # Add edge u-v
            u = int(qry[1])
            v = int(qry[2])
            add_count += 1

            # push v into u's list
            nb[ptr] = v
            tadd[ptr] = add_count
            nxt[ptr] = head[u]
            head[u] = ptr
            ptr += 1

            # push u into v's list
            nb[ptr] = u
            tadd[ptr] = add_count
            nxt[ptr] = head[v]
            head[v] = ptr
            ptr += 1

            # update degrees and isolated count
            if deg[u] == 0:
                isolated -= 1
            deg[u] += 1
            if deg[v] == 0:
                isolated -= 1
            deg[v] += 1

        else:
            # Clear all edges incident to v
            v = int(qry[1])
            # if v was non‐isolated, it becomes isolated
            if deg[v] > 0:
                isolated += 1
                deg[v] = 0
            # traverse v's adjacency list and decrement neighbors
            cur = head[v]
            head[v] = 0
            while cur != 0:
                u = nb[cur]
                ta = tadd[cur]
                # only remove if u hasn't been cleared after this edge was added
                if clear_time[u] < ta:
                    deg[u] -= 1
                    if deg[u] == 0:
                        isolated += 1
                cur = nxt[cur]
            # mark clear time of v
            clear_time[v] = add_count

        out.append(str(isolated))

    # output all answers
    sys.stdout.write("
".join(out))

# call main
main()