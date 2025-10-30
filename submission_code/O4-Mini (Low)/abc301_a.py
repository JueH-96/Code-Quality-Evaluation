def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()

    # Count total wins
    t_total = S.count('T')
    a_total = S.count('A')

    # If one has strictly more, that's the winner
    if t_total > a_total:
        print('T')
        return
    elif a_total > t_total:
        print('A')
        return

    # Otherwise they are tied; we find who reached the final count first
    target = t_total  # same as a_total
    t_cum = 0
    a_cum = 0
    for c in S:
        if c == 'T':
            t_cum += 1
        else:
            a_cum += 1

        # Check if someone just reached the target
        if t_cum == target:
            print('T')
            return
        if a_cum == target:
            print('A')
            return

if __name__ == "__main__":
    main()