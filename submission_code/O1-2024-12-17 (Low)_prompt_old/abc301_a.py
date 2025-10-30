def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Count total wins for T and A
    t_total = S.count('T')
    a_total = S.count('A')

    # If there's a clear winner by total wins
    if t_total > a_total:
        print('T')
        return
    if a_total > t_total:
        print('A')
        return

    # If they are tied, find who reached the final tied number first
    needed = t_total  # same as a_total
    t_count, a_count = 0, 0
    for char in S:
        if char == 'T':
            t_count += 1
        else:
            a_count += 1
        if t_count == needed:
            print('T')
            return
        if a_count == needed:
            print('A')
            return

# Call solve()
solve()