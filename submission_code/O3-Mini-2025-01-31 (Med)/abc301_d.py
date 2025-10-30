def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]
    N = int(data[1])
    
    L = len(S)
    # If N is at least 2^L, then any number formed from S (which is at most 2^L-1) is ≤ N.
    # In that case, the best answer is simply the maximum value from T,
    # which is obtained by replacing every '?' with '1'.
    if N >= (1 << L):
        max_str = ''.join('1' if ch == '?' else ch for ch in S)
        print(int(max_str, 2))
        return
    
    # Otherwise, we must ensure the binary number formed from S (exactly L bits)
    # does not exceed N. To do that we use dynamic programming to fill the '?'.
    # We first represent N as a binary string of length L (with leading zeros).
    target = format(N, '0{}b'.format(L))
    
    # dp(i, tight) returns the maximum integer value that can be formed using S[i:]
    # such that the prefix is consistent with the tight bound.
    # "tight" = True means that so far we have matched the prefix of 'target' exactly.
    # When False, we are already strictly less so remaining positions can be chosen greedily.
    # Returns None if it is impossible to form a valid number.
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(i, tight):
        if i == L:
            return 0  # Base case: no remaining bits contribute 0.
        
        best = None
        # If the current character is fixed (not a '?'):
        if S[i] != '?':
            d = int(S[i])
            if tight:
                tdig = int(target[i])
                if d > tdig:
                    # This fixed digit exceeds the target at this position: reject.
                    pass
                else:
                    # If it equals, we remain in "tight" state; if less, we relax the constraint.
                    new_tight = (d == tdig)
                    remainder = dp(i + 1, new_tight)
                    if remainder is not None:
                        value = d * (1 << (L - i - 1)) + remainder
                        best = value
            else:
                remainder = dp(i + 1, False)
                if remainder is not None:
                    value = d * (1 << (L - i - 1)) + remainder
                    best = value
        else:
            # S[i] is '?': try replacing it with 1 or 0.
            # We try 1 first because we want to maximize the number.
            for d in [1, 0]:
                if tight:
                    tdig = int(target[i])
                    if d > tdig:
                        continue
                    new_tight = (d == tdig)
                else:
                    new_tight = False
                remainder = dp(i + 1, new_tight)
                if remainder is not None:
                    value = d * (1 << (L - i - 1)) + remainder
                    if best is None or value > best:
                        best = value
        return best

    answer = dp(0, True)
    # If no valid assignment exists or no value ≤ N is found, print -1.
    if answer is None:
        print(-1)
    else:
        print(answer)

if __name__ == '__main__':
    main()