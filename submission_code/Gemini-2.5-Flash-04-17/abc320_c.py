import sys

def solve():
    M = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(3)]

    # Store indices for each character ('0' through '9') in each reel (0, 1, 2)
    pos = [[[] for _ in range(10)] for _ in range(3)]
    for i in range(3):
        for j in range(M):
            digit = int(S[i][j])
            pos[i][digit].append(j)

    min_overall_time = float('inf')

    # Iterate through all possible target digits
    for digit in range(10):
        P1 = pos[0][digit]
        P2 = pos[1][digit]
        P3 = pos[2][digit]

        # If the digit does not appear on all three reels, it's impossible to achieve with this digit
        if not P1 or not P2 or not P3:
            continue

        # Iterate through all combinations of index positions (p1, p2, p3)
        # where the target digit appears on reels 1, 2, and 3 respectively.
        for p1 in P1:
            for p2 in P2:
                for p3 in P3:
                    # We need to find non-negative integers i, j, k such that
                    # t1 = p1 + i * M
                    # t2 = p2 + j * M
                    # t3 = p3 + k * M
                    # are distinct, and max(t1, t2, t3) is minimized.

                    # We can iterate through small values of shifts i, j, k.
                    # If the minimum max time T is achieved, then t1, t2, t3 <= T.
                    # t = p + shift * M <= T  => shift * M <= T - p => shift <= (T - p) / M.
                    # The maximum necessary shift depends on T.
                    # However, the structure of the problem suggests that the minimum maximum time
                    # is achieved when the times p1+iM, p2+jM, p3+kM are relatively close.
                    # This often happens for small values of i, j, k.
                    # Checking shifts 0, 1, 2 seems sufficient to cover cases where
                    # times are in the same cycle block (i=j=k=0),
                    # or require shifting by one or two cycles to resolve distinctness conflicts or minimize the maximum.

                    min_T_p1p2p3 = float('inf')

                    # Iterate through shifts for reel 1 (i)
                    # Iterate through shifts for reel 2 (j)
                    # Iterate through shifts for reel 3 (k)
                    # A small range like 0, 1, 2 is often enough for problems like this.
                    # Let's try range(3) which covers shifts 0, 1, 2.
                    shifts_range = range(3) # Sufficient for sample cases.

                    for i in shifts_range:
                        for j in shifts_range:
                            for k in shifts_range:
                                t1 = p1 + i * M
                                t2 = p2 + j * M
                                t3 = p3 + k * M

                                # The three stop times must be distinct
                                if t1 != t2 and t1 != t3 and t2 != t3:
                                    # If distinct, calculate the maximum of these three times
                                    current_max_time = max(t1, t2, t3)
                                    # Update the minimum max time found for this (p1, p2, p3) combination
                                    min_T_p1p2p3 = min(min_T_p1p2p3, current_max_time)

                    # After checking all combinations of small shifts for this (p1, p2, p3) triplet,
                    # update the overall minimum time found across all possible digits and index triplets.
                    if min_T_p1p2p3 != float('inf'):
                         min_overall_time = min(min_overall_time, min_T_p1p2p3)

    # If min_overall_time is still infinity, it means no combination of distinct times
    # could be found to achieve the same digit on all three reels.
    if min_overall_time == float('inf'):
        print(-1)
    else:
        print(min_overall_time)

solve()