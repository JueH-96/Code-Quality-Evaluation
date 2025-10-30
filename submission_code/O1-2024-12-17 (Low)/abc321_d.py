def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    P = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))

    # Step 1: Sort A and B
    A.sort()
    B.sort()

    # Step 2: Create prefix sums for B
    prefixB = [0] * (M+1)
    for i in range(M):
        prefixB[i+1] = prefixB[i] + B[i]

    # Step 3: For each A_i, find how many B_j satisfy A_i + B_j <= P
    #         Then accumulate the appropriate sum.
    ans = 0
    for ai in A:
        limit = P - ai
        # idx is the count of B_j that satisfy B_j <= limit
        # using bisect_right to find the insertion point
        idx = bisect.bisect_right(B, limit)
        # Sum of those valid pairs: sum(A_i + B_j) = idx*A_i + sum(B_j for j in those idx)
        ans += ai * idx + prefixB[idx]
        # For the remaining (M - idx) pairs, the price is P
        ans += (M - idx) * P

    print(ans)

# Don't forget to call main() at the end
if __name__ == "__main__":
    main()