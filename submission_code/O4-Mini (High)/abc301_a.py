def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    S = data[1]
    # Count final wins
    t_total = S.count('T')
    a_total = N - t_total
    # If one has more wins, that's the overall winner
    if t_total > a_total:
        print('T')
        return
    if a_total > t_total:
        print('A')
        return
    # Otherwise they are tied: find who reached that tied count first
    needed = t_total  # same as a_total
    t_count = 0
    a_count = 0
    for ch in S:
        if ch == 'T':
            t_count += 1
            if t_count == needed:
                print('T')
                return
        else:
            a_count += 1
            if a_count == needed:
                print('A')
                return

if __name__ == "__main__":
    main()