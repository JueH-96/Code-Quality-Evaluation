import sys
def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    t = [0]*N
    x = [0]*N
    for i in range(N):
        ti, xi = map(int, input().split())
        t[i] = ti
        x[i] = xi

    # Assign an ID to each pickup event in order
    pu_id = [-1]*N
    total_pu = 0
    # For each type, record lists of pickup positions and their pu_ids,
    # and list of demand (monster) positions.
    pickups_by_type = [[] for _ in range(N+1)]
    pu_ids_by_type = [[] for _ in range(N+1)]
    demands_by_type = [[] for _ in range(N+1)]

    for i in range(N):
        if t[i] == 1:
            pu_id[i] = total_pu
            pickups_by_type[x[i]].append(i)
            pu_ids_by_type[x[i]].append(total_pu)
            total_pu += 1
        else:
            demands_by_type[x[i]].append(i)

    # Decision array for pickups: 1 = pick, 0 = skip
    pick_dec = [0]*total_pu

    # For each type, match demands to the latest possible preceding pickups
    for typ in range(1, N+1):
        demands = demands_by_type[typ]
        if not demands:
            continue
        pickups = pickups_by_type[typ]
        pu_ids = pu_ids_by_type[typ]
        # Impossible if fewer pickups than demands
        if len(pickups) < len(demands):
            print(-1)
            return
        # Greedily match from the last demand backwards
        k = len(pickups) - 1
        for dpos in reversed(demands):
            # Move k to a pickup strictly before the demand
            while k >= 0 and pickups[k] >= dpos:
                k -= 1
            if k < 0:
                # no pickup available before this demand
                print(-1)
                return
            # Use that pickup
            pick_dec[pu_ids[k]] = 1
            k -= 1

    # Simulate to find the peak inventory K
    inv = 0
    K = 0
    for i in range(N):
        if t[i] == 1:
            # If this pickup is chosen, inventory increases
            if pick_dec[pu_id[i]] == 1:
                inv += 1
                if inv > K:
                    K = inv
        else:
            # A monster is defeated, inventory decreases
            inv -= 1

    # Output the result
    # First line: the minimum possible peak K
    # Second line: for each pickup event in order, 1 or 0
    print(K)
    if total_pu > 0:
        res = []
        for i in range(N):
            if t[i] == 1:
                res.append(str(pick_dec[pu_id[i]]))
        print(" ".join(res))
    else:
        # No pickups at all: print an empty line
        print()

if __name__ == "__main__":
    main()