def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Count total wins for each
    t_wins = S.count('T')
    a_wins = S.count('A')

    if t_wins > a_wins:
        print('T')
        return
    elif a_wins > t_wins:
        print('A')
        return
    else:
        # If tied, find who reached final_count first
        final_count = t_wins  # same as a_wins
        t_in_progress = 0
        a_in_progress = 0
        for c in S:
            if c == 'T':
                t_in_progress += 1
                if t_in_progress == final_count:
                    print('T')
                    return
            else:  # c == 'A'
                a_in_progress += 1
                if a_in_progress == final_count:
                    print('A')
                    return

# Do not remove. This is required to run the solution.
if __name__ == "__main__":
    main()