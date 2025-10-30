import sys

def main():
    n, k = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    p = [x - 1 for x in p]  # convert to 0-based

    visited = [False] * n
    cycles = []
    cycle_id = [0] * n
    pos_in_cycle = [0] * n
    cycle_length = [0] * n

    for i in range(n):
        if not visited[i]:
            current = i
            new_cycle = []
            while True:
                if visited[current]:
                    break
                new_cycle.append(current)
                visited[current] = True
                current = p[current]
            cycles.append(new_cycle)
            # Assign cycle info for each element in new_cycle
            for idx in range(len(new_cycle)):
                elem = new_cycle[idx]
                cid = len(cycles) - 1
                cycle_id[elem] = cid
                pos_in_cycle[elem] = idx
                cycle_length[elem] = len(new_cycle)

    new_p = [0] * n
    for i in range(n):
        L = cycle_length[i]
        rem = k % L
        pos = pos_in_cycle[i] + rem
        pos %= L
        cid = cycle_id[i]
        cycle_list = cycles[cid]
        new_val = cycle_list[pos]
        new_p[i] = new_val

    print(' '.join(str(x + 1) for x in new_p))

if __name__ == "__main__":
    main()