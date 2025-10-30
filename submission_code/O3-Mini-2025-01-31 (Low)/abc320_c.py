def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    M = int(data[0])
    S1 = data[1]
    S2 = data[2]
    S3 = data[3]
    reels = [S1, S2, S3]

    # For each digit d from '0' to '9'
    # For each reel, we want to find the smallest t (non-negative integer) such that
    # t mod M equals one of the positions at which the reel's symbol equals d.
    # Because valid times for an occurrence at position pos are all numbers of the form pos + k*M (k >= 0),
    # the earliest time we can stop a reel showing d is simply the minimum index pos
    # such that the character at that index equals d.
    # Once we do that for all three reels, if every reel has at least one occurrence of d,
    # then if we schedule the stops at those minimal times, the global time (when the last stop happens)
    # is the maximum among the three. Clearly, waiting any longer wouldn’t help reduce the maximum.
    # Thus, for each d that occurs in all reels we compute candidate = max(min_time for each reel).
    # And we take the minimum candidate over all d.
    answer = None
    for d in "0123456789":
        min_times = []
        valid = True
        for reel in reels:
            # find the earliest position (which is the smallest t possible) where reel equals d.
            found = False
            for i, ch in enumerate(reel):
                if ch == d:
                    min_times.append(i)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            candidate = max(min_times)
            if answer is None or candidate < answer:
                answer = candidate

    if answer is None:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(answer))


if __name__ == '__main__':
    main()