def main():
    import sys
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    
    # Helper to interpret the string S by substituting all '?' with either '0' or '1'.
    # fill is a character ('0' or '1') to use in place of '?'
    def interpret(pattern, fill):
        val = 0
        for ch in pattern:
            if ch == '0':
                val = (val << 1)
            elif ch == '1':
                val = (val << 1) + 1
            else:  # ch == '?'
                val = (val << 1) + (1 if fill == '1' else 0)
        return val

    # Compute the minimum and maximum possible values from S
    x_min = interpret(S, '0')
    x_max = interpret(S, '1')

    # If even the smallest possible number is too large, no valid answer
    if x_min > N:
        print(-1)
        return

    # Determine the bit-length of N
    B = len(bin(N)) - 2  # subtract 2 for the "0b" prefix
    M = len(S)

    # If N has strictly more bits than S, then x_max (the largest possible S) is surely ≤ N
    if B > M:
        print(x_max)
        return

    # If the largest possible S is already ≤ N, we're done
    if x_max <= N:
        print(x_max)
        return

    # Otherwise, we need to find the best fit ≤ N via DP.

    # Convert N into a bit-array of length M (left padded with zeros if needed)
    n_bits = bin(N)[2:]  # binary form of N as a string, most significant bit first
    n = [0]*M
    offset = M - len(n_bits)
    for i in range(M):
        if i < offset:
            n[i] = 0
        else:
            n[i] = int(n_bits[i - offset])

    # We'll memoize dp states:
    # dp(i, less) → largest decimal value we can form from S[i..] subject to
    # (less == 1 if we've already placed a smaller prefix than N's prefix, else 0)
    memo = [[None]*2 for _ in range(M+1)]

    def dp(i, less):
        # If we've assigned all bits, return 0 as there's nothing more to add
        if i == M:
            return 0
        # If we have a cached result, return it
        if memo[i][less] is not None:
            return memo[i][less]

        b = n[i]  # corresponding bit of N at position i
        ch = S[i]

        if ch == '?':
            possible_bits = [0, 1]
        else:
            possible_bits = [int(ch)]

        best_val = None

        for c in possible_bits:
            # If not already smaller, we cannot exceed the corresponding N bit b
            if less == 0 and c > b:
                continue

            new_less = less or (c < b)
            remainder = dp(i+1, 1 if new_less else 0)
            if remainder is not None:
                candidate = (c << (M-1 - i)) + remainder
                if best_val is None or candidate > best_val:
                    best_val = candidate

        memo[i][less] = best_val
        return best_val

    ans = dp(0, 0)
    if ans is None:
        print(-1)
    else:
        print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()