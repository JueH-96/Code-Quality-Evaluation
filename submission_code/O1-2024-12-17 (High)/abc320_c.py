def main():
    import sys
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    S1 = data[1]
    S2 = data[2]
    S3 = data[3]

    # For each reel i and digit d, store all positions j where S_i[j] == d
    pos1 = [[] for _ in range(10)]
    pos2 = [[] for _ in range(10)]
    pos3 = [[] for _ in range(10)]
    for i in range(M):
        d1 = ord(S1[i]) - ord('0')
        d2 = ord(S2[i]) - ord('0')
        d3 = ord(S3[i]) - ord('0')
        pos1[d1].append(i)
        pos2[d2].append(i)
        pos3[d3].append(i)

    INF = 10**15
    best = INF

    # Check each possible digit (0 through 9)
    for digit in range(10):
        # If any reel cannot show this digit at all, skip
        if not pos1[digit] or not pos2[digit] or not pos3[digit]:
            continue

        # Try all combinations of remainders x in Reel1, y in Reel2, z in Reel3
        for x in pos1[digit]:
            for y in pos2[digit]:
                # T1 corresponds to remainder x on reel1
                # T2 corresponds to remainder y on reel2 (shift if needed to avoid T1)
                if x == y:
                    T1, T2 = x, y + M
                else:
                    T1, T2 = x, y

                # Precompute the partial max for T1 and T2
                max_T1_T2 = T1 if T1 > T2 else T2

                # Now try to find a valid T3 for remainder z on reel3
                for z in pos3[digit]:
                    T3 = z
                    # Ensure T3 is distinct from T1 and T2 by shifting up to 2*M
                    if T3 == T1 or T3 == T2:
                        T3 += M
                        if T3 == T1 or T3 == T2:
                            T3 += M
                            # If still collides, no valid way for this triple
                            if T3 == T1 or T3 == T2:
                                continue

                    current_max = max_T1_T2 if max_T1_T2 > T3 else T3
                    if current_max < best:
                        best = current_max

    # If best was never updated, print -1; otherwise print the minimum possible time
    print(-1 if best == INF else best)

# Remember to call main() at the end
if __name__ == "__main__":
    main()