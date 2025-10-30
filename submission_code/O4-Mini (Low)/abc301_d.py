import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    M = len(S)
    # If N >= 2^M, any M-bit string is <= N
    if N >= (1 << M):
        # Just take max: replace '?' with '1'
        res_bits = ['0'] * M
        for i, ch in enumerate(S):
            if ch == '1':
                res_bits[i] = '1'
            elif ch == '?':
                res_bits[i] = '1'
            # else '0' stays
        # parse
        val = 0
        for b in res_bits:
            val = (val << 1) | (1 if b == '1' else 0)
        print(val)
        return

    # Build N's binary representation, pad to length M
    nbits = [0] * M
    binN = bin(N)[2:].rjust(M, '0')
    for i, ch in enumerate(binN):
        nbits[i] = 1 if ch == '1' else 0

    from functools import lru_cache

    @lru_cache(None)
    def dp(i, tight):
        # returns a tuple of bits (as characters) for positions i..M-1, or None if impossible
        if i == M:
            return ()  # empty suffix
        # allowed bits at S[i]
        allowed = []
        if S[i] == '0':
            allowed = [0]
        elif S[i] == '1':
            allowed = [1]
        else:  # '?'
            allowed = [0, 1]
        # maximum first, so try b=1 then 0
        for b in (1, 0):
            if b not in allowed:
                continue
            if tight:
                if b > nbits[i]:
                    continue
                ntight = (b == nbits[i])
            else:
                ntight = False
            suffix = dp(i+1, ntight)
            if suffix is not None:
                # found valid
                return (str(b),) + suffix
        return None

    ans_bits = dp(0, True)
    if ans_bits is None:
        print(-1)
    else:
        val = 0
        for c in ans_bits:
            val = (val << 1) | (1 if c == '1' else 0)
        print(val)

if __name__ == "__main__":
    main()