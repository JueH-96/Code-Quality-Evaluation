def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # First line: N, T
    # Then N lines for the strings S_i
    # input_data[0] = N, input_data[1] = T, then S_1..S_N follow

    N = int(input_data[0])
    T = input_data[1]
    S_list = input_data[2:]  # the N strings

    # To handle edge cases quickly:
    # if T is empty (not stated in the problem, but just in case),
    # every pair (i, j) trivially matches. But per constraints, len(T) >= 1.
    # We'll proceed as normal.

    T_rev = T[::-1]
    M = len(T)

    # Function to find how many characters of T (from start) 
    # can be matched as a subsequence in s.
    def match_prefix(s, T):
        p = 0
        m = len(T)
        for c in s:
            if T[p] == c:
                p += 1
                if p == m:
                    break
        return p

    # Function to find how many characters of T_rev (from start)
    # can be matched as a subsequence in reversed(s),
    # i.e. how large a suffix of T is matched by s.
    def match_suffix(s, T_rev):
        p = 0
        m = len(T_rev)
        # iterate over s in reverse
        for c in reversed(s):
            if T_rev[p] == c:
                p += 1
                if p == m:
                    break
        return p

    A = [0]*N  # how many of T's chars each S_i can match from the front
    B = [0]*N  # how many of T_rev's chars each S_i can match from the front (== suffix of T)

    # Compute A[i], B[i] for each S_i
    idx = 0
    for i in range(N):
        s = S_list[i]
        A[i] = match_prefix(s, T)
        B[i] = match_suffix(s, T_rev)

    # Sort B for two-pointer / binary search approach
    B.sort()

    # We want to count the number of pairs (i, j) where A[i] + B[j] >= M
    # For each a in A, find how many b in B satisfy b >= M - a
    # This is the same as counting how many b are >= X = (M - a).
    # Use bisect_left to find the first index where B[idx] >= X.
    # The count is len(B) - idx.

    ans = 0
    n = len(B)
    for a in A:
        need = M - a  # b must be at least this
        # If need <= 0, all B qualify
        if need <= 0:
            ans += n
        else:
            pos = bisect.bisect_left(B, need)
            ans += (n - pos)

    print(ans)

# Do not forget to call main!
if __name__ == "__main__":
    main()