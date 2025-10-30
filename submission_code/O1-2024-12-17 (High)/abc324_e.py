def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    T = data[1]
    M = len(T)
    
    # We will compute, for each string S_i:
    # 1. prefix[i]: how many characters from the start of T can be matched in S_i 
    #    when scanning both from left to right.
    # 2. suffix[i]: how many characters from the end of T can be matched in S_i 
    #    when scanning S_i from right to left, and T from right to left.
    #
    # Then, a pair (i, j) satisfies that T is a subsequence of S_i + S_j if and only if
    # there is some split x in T (0 <= x <= M) such that T[:x] can be matched in S_i
    # and T[x:] can be matched in S_j. This is equivalent to having:
    #     prefix[i] + suffix[j] >= M
    #
    # Explanation: If prefix[i] is the maximum length of T's prefix matched by S_i, 
    # then T[:k] is also matched for any k <= prefix[i]. Similarly, if suffix[j] = r, 
    # then T[M-r:] is matched by S_j. We just need some k = M - r <= prefix[i]. This 
    # condition is exactly prefix[i] + suffix[j] >= M.

    prefix_vals = [0]*N
    suffix_vals = [0]*N

    # Function to compute how many characters from T's start can be matched
    # in a string S by scanning both left to right (standard two-pointer subsequence).
    def get_prefix(S, T):
        p = 0
        for c in S:
            if c == T[p]:
                p += 1
                if p == len(T):
                    break
        return p

    # Function to compute how many characters from T's end can be matched
    # in a string S by scanning S right to left, and T right to left (another standard
    # two-pointer approach).
    def get_suffix(S, T):
        p = len(T) - 1
        for c in reversed(S):
            if c == T[p]:
                p -= 1
                if p < 0:
                    break
        return len(T) - 1 - p

    # Read each S_i from data (after N,T which are data[0], data[1]).
    idx = 2
    for i in range(N):
        S = data[idx]
        idx += 1
        prefix_vals[i] = get_prefix(S, T)
        suffix_vals[i] = get_suffix(S, T)

    # Sort the prefix and suffix arrays
    prefix_vals.sort()
    suffix_vals.sort()

    # We'll count how many (i, j) satisfy prefix[i] + suffix[j] >= M
    # Standard two-pointer method for "count pairs with sum >= X" in sorted arrays:
    i, j = 0, N-1
    count = 0
    while i < N and j >= 0:
        if prefix_vals[i] + suffix_vals[j] >= M:
            # All indices between i..(N-1) with this j satisfy the condition.
            count += (N - i)
            j -= 1
        else:
            i += 1

    print(count)

# Don't forget to call main!
if __name__ == "__main__":
    main()