def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353

    # Read N, M
    N, M = map(int, input_data[:2])
    ptr = 2

    # We will collect all constraints by (l, r) -> set of forbidden positions.
    # Because multiple constraints can share the same (l, r) but forbid different X_i.
    from collections import defaultdict
    interval_forbidden = defaultdict(set)

    # Read each constraint
    for _ in range(M):
        l = int(input_data[ptr]); r = int(input_data[ptr+1]); x = int(input_data[ptr+2])
        ptr += 3
        interval_forbidden[(l,r)].add(x)

    # Convert the dictionary to a list of intervals
    # Each element will be (l, r, forbidden_set, active_flag)
    intervals = []
    for (l, r), forbset in interval_forbidden.items():
        # We'll store them as (l, r, list_of_forbidden_positions, active_flag)
        # active_flag indicates if this interval is still "unresolved" (hasn't yet found its max).
        intervals.append([l, r, list(forbset), True])

    # Precompute, for each x in [1..N], all intervals that contain x.
    # This is the main part that can be large (up to M*N), but N<=500 should be (500*1e5)=5e7 in worst case,
    # which is borderline but we will attempt to do it carefully and as efficiently as possible.
    # We also need to compute how many active intervals forbid x.
    # Let's keep a list of intervals_containing[x] = list of indices in 'intervals'.
    intervals_containing = [[] for _ in range(N+1)]

    # active_forbid_count[x]: how many active intervals forbid x
    active_forbid_count = [0]*(N+1)

    # Build intervals_containing and initialize active_forbid_count
    # We'll do this by going through each interval once and adding the index to intervals_containing[x]
    # for x in [l..r]. Then for each x in the forbidden set, increment active_forbid_count[x].
    # To avoid a python range() overhead for large (r-l) in a naive loop, we can do it directly:
    # since r-l <= 499 at most (because N<=500), a simple for x in range(l, r+1) is acceptable.

    for i, (l, r, fset, _) in enumerate(intervals):
        for x in range(l, r+1):
            intervals_containing[x].append(i)
        for x in fset:
            active_forbid_count[x] += 1  # x is forbidden for this interval

    # assigned[x] indicates if position x has already been assigned a (larger) label
    assigned = [False]*(N+1)

    answer = 1

    # We'll assign labels from largest (N) down to smallest (1).
    # At step k (from N down to 1), we look for those x not yet assigned
    # and active_forbid_count[x] == 0. The number of such x is how many ways
    # we can place the label k. We multiply answer by that count.
    # Then we "pick" one of them (arbitrarily for the sake of updating structures),
    # and mark all intervals that contain that x as resolved ("active=False").
    # For each such interval, for each y in its forbidden list, we decrement active_forbid_count[y].
    # This simulates that once an interval is resolved (its max found among bigger labels), 
    # it no longer forbids anything.

    for label in range(N, 0, -1):
        # Collect all valid positions
        valid_positions = []
        for x in range(1, N+1):
            if (not assigned[x]) and (active_forbid_count[x] == 0):
                valid_positions.append(x)

        c = len(valid_positions)
        if c == 0:
            print(0)
            return

        answer = (answer * c) % MOD

        # Now pick one position (say the last one) to actually assign label=label.
        chosen_x = valid_positions[-1]
        assigned[chosen_x] = True

        # Deactivate intervals that contain chosen_x
        for i in intervals_containing[chosen_x]:
            if intervals[i][3]:  # if still active
                # intervals[i]: (l, r, forb_list, active_flag)
                forb_list = intervals[i][2]
                # For each y in forb_list, decrement active_forbid_count[y] if not assigned
                for y in forb_list:
                    if not assigned[y]:
                        active_forbid_count[y] -= 1
                intervals[i][3] = False  # mark interval as resolved

    print(answer % MOD)