def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    S = input().strip()

    ans = 0
    cur1 = 0  # count of '1's in current segment
    cur2 = 0  # count of '2's in current segment

    for c in S:
        if c == '0':
            # end of a segment: compute how many logo T-shirts needed here
            # need one logo for each '2', and for '1's beyond M plain shirts
            needed = cur2 + max(0, cur1 - M)
            ans = max(ans, needed)
            cur1 = cur2 = 0
        elif c == '1':
            cur1 += 1
        else:  # c == '2'
            cur2 += 1

    # handle last segment if it doesn't end with '0'
    needed = cur2 + max(0, cur1 - M)
    ans = max(ans, needed)

    print(ans)

if __name__ == "__main__":
    main()