import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    T = int(input())
    # We will record, for each couple a, the two positions where a appears:
    # l[a] = first occurrence index (1‐based)
    # r[a] = second occurrence index
    # The only way two non‐adjacent couples a,b can both be made adjacent
    # by swapping only between a and b seats is when their endpoints are
    # _pairwise_ consecutive in the lineup, i.e.
    #   l[b] == l[a] + 1  and  r[b] == r[a] + 1
    # (with a<b).  That guarantees that among the four seats {l[a],l[b],r[a],r[b]}
    # there are two disjoint adjacent pairs, so we can assign one pair to a
    # and one to b by swapping.  This exactly matches the examples.
    #
    # So for each test case, we:
    # 1) Read N and the array A of length 2N.
    # 2) Build l[1..N], r[1..N].
    # 3) Count how many a in [1..N-1] satisfy
    #       l[a+1] == l[a] + 1  and  r[a+1] == r[a] + 1
    #    but also ensure that neither couple was originally adjacent:
    #       r[a] > l[a] + 1   (non‐adjacent couple a)
    #       r[a+1] > l[a+1] + 1 (non‐adjacent couple a+1)
    #
    # Note: If a couple were already adjacent, we must not count any pairs
    # involving that couple.

    out = []
    for _ in range(T):
        N = int(input())
        A = list(map(int, input().split()))
        # record occurrences
        l = [0]*(N+1)
        r = [0]*(N+1)
        for i, x in enumerate(A, start=1):
            if l[x] == 0:
                l[x] = i
            else:
                r[x] = i

        ans = 0
        # scan a=1..N-1
        for a in range(1, N):
            # non‐adjacent checks
            if r[a] == l[a] + 1:          # couple a already adjacent
                continue
            if r[a+1] == l[a+1] + 1:      # couple a+1 already adjacent
                continue
            # endpoint adjacency
            if l[a+1] == l[a] + 1 and r[a+1] == r[a] + 1:
                ans += 1

        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()