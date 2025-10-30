def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Count the number of wins for Takahashi (T) and Aoki (A)
    count_t = S.count('T')
    count_a = S.count('A')

    if count_t > count_a:
        # Takahashi wins outright
        print('T')
    elif count_a > count_t:
        # Aoki wins outright
        print('A')
    else:
        # Tie in the total number of wins, determine who reached that number first
        needed_wins = count_t  # same as count_a, since count_t = count_a
        tak_wins_so_far, aok_wins_so_far = 0, 0
        for ch in S:
            if ch == 'T':
                tak_wins_so_far += 1
                if tak_wins_so_far == needed_wins:
                    print('T')
                    return
            else:
                aok_wins_so_far += 1
                if aok_wins_so_far == needed_wins:
                    print('A')
                    return

# Do not forget to call the main function
main()