def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()

    total_T = S.count("T")
    total_A = S.count("A")

    # If one player won more games, they are the overall winner.
    if total_T != total_A:
        print("T" if total_T > total_A else "A")
        return

    # When both have the same number of wins, the winner is determined by who reached that win count first.
    target = total_T  # This is equal to total_A in this case.
    count_T, count_A = 0, 0
    for ch in S:
        if ch == "T":
            count_T += 1
        else:  # ch is "A"
            count_A += 1
        if count_T == target and count_A < target:
            print("T")
            return
        if count_A == target and count_T < target:
            print("A")
            return

if __name__ == "__main__":
    main()