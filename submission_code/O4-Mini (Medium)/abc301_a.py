def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    S = input().strip()

    # Total wins
    t_wins = S.count('T')
    a_wins = N - t_wins

    # If one has more wins, that's the overall winner
    if t_wins > a_wins:
        print('T')
        return
    if a_wins > t_wins:
        print('A')
        return

    # Otherwise they are tied; find who reached that tie‐winning count first.
    target = t_wins  # == a_wins
    t_count = 0
    a_count = 0
    for ch in S:
        if ch == 'T':
            t_count += 1
        else:
            a_count += 1

        # The first to reach the target wins
        if t_count == target:
            print('T')
            return
        if a_count == target:
            print('A')
            return

if __name__ == "__main__":
    main()