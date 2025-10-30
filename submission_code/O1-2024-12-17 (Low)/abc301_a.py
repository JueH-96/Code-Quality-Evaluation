def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Count the total wins of T and A
    t_total = S.count('T')
    a_total = S.count('A')

    # If one has more wins, that one is the overall winner
    if t_total > a_total:
        print('T')
    elif a_total > t_total:
        print('A')
    else:
        # Tie case: check who reached the final count first
        needed = t_total  # t_total == a_total in case of tie
        t_count, a_count = 0, 0
        
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

# Do not forget to call main
if __name__ == "__main__":
    main()