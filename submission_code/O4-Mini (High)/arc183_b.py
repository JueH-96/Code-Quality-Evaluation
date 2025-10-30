import sys
import threading

def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin
    readline = data.readline

    T_line = readline().strip()
    if not T_line:
        return
    T = int(T_line)
    out_lines = []
    for _ in range(T):
        line = readline().split()
        # skip empty lines
        while not line:
            line = readline().split()
        N = int(line[0]); K = int(line[1])
        A = list(map(int, readline().split()))
        B = list(map(int, readline().split()))

        # Collect positions where B_i = v
        BPositions = {}
        for idx, v in enumerate(B, start=1):
            if v in BPositions:
                BPositions[v].append(idx)
            else:
                BPositions[v] = [idx]

        # Collect root positions where A_j = v for v that appear in B
        RootPositions = {}
        # We only need roots for v that appear in BPositions
        # If A_j value not in BPositions, we don't need it
        for j, v in enumerate(A, start=1):
            if v in BPositions:
                if v in RootPositions:
                    RootPositions[v].append(j)
                else:
                    RootPositions[v] = [j]

        possible = True
        # For each value v that appears in B
        for v, P in BPositions.items():
            # list of B positions for v
            P.sort()
            # list of root positions for v
            RP = RootPositions.get(v)
            if not RP:
                # no initial v at all => fail
                possible = False
                break
            RP.sort()
            # scan segments in P where consecutive gap > K
            seg_start = P[0]
            prev = seg_start
            # helper to check a segment [l..r]
            def check_segment(l, r):
                # need at least one root j with |j - l| <= K or |j - r| <= K
                # equivalently j in [l-K, r+K]
                low = l - K
                if low < 1:
                    low = 1
                high = r + K
                if high > N:
                    high = N
                # binary search in RP for first >= low
                i = bisect_left(RP, low)
                if i == len(RP) or RP[i] > high:
                    return False
                return True

            # iterate through P to break into segments
            for pos in P[1:]:
                if pos - prev > K:
                    # end segment at prev
                    if not check_segment(seg_start, prev):
                        possible = False
                        break
                    # start new segment
                    seg_start = pos
                prev = pos
            if not possible:
                break
            # last segment
            if not check_segment(seg_start, prev):
                possible = False
                break

        out_lines.append("Yes" if possible else "No")

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()