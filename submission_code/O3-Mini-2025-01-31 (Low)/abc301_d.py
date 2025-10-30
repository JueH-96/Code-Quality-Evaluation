def main():
    import sys
    sys.setrecursionlimit(10000)
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    S = data[0].strip()
    N = int(data[1])
    L = len(S)
    
    # First, compute the maximum value from S by replacing each '?' with '1'
    max_val = 0
    for ch in S:
        max_val = (max_val << 1) + (1 if ch in "1?" else 0)
    
    # If this maximum value is <= N, then it's the answer.
    if max_val <= N:
        sys.stdout.write(str(max_val))
        return

    # Also, note that if N is >= 2^L, then all numbers represented by S (with L bits) are <= N,
    # and hence max_val (which is the largest among them) would be the answer.
    if (1 << L) - 1 <= N:
        sys.stdout.write(str(max_val))
        return

    # Otherwise, we need to choose assignments for '?' such that the resulting value is <= N,
    # and it is maximum possible.
    # We view S as defining a fixed-length binary number with possible free bits.
    # To check against N, we represent N in binary with L digits (padded with leading zeros if necessary).
    N_bin = format(N, '0{}b'.format(L))
    
    # We'll use memoized recursion (DP) to try choices from left to right.
    # State: position i and flag 'tight'
    # tight = True means so far we have exactly matched the prefix of N_bin.
    # If tight is False we are free to choose the maximum bits.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(i, tight):
        if i == L:
            return ""  # reached end; return empty string as suffix.
        # Determine available bit if S[i] is fixed or free.
        choices = []
        if S[i] == '?':
            # To maximize the number, try '1' first then '0'
            choices = [1, 0]
        else:
            choices = [int(S[i])]
        
        for bit in choices:
            current_n_bit = int(N_bin[i])
            # If we are tight and our chosen bit exceeds that of N, it's not valid.
            if tight and bit > current_n_bit:
                continue
            # Update the tight flag: remains True only if this bit equals N_bin[i] and tight was True.
            new_tight = tight and (bit == current_n_bit)
            rest = dp(i + 1, new_tight)
            if rest is not None:
                # Return the current bit concatenated with the valid suffix.
                return str(bit) + rest
        return None  # if no valid assignment, return None

    result = dp(0, True)
    if result is None:
        sys.stdout.write(str(-1))
    else:
        sys.stdout.write(str(int(result, 2)))
        
if __name__ == '__main__':
    main()