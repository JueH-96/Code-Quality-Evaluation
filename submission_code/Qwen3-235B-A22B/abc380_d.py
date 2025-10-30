def main():
    import sys
    S = sys.stdin.readline().strip()
    L = len(S)
    Q = int(sys.stdin.readline())
    Ks = list(map(int, sys.stdin.readline().split()))
    ans = []
    for K in Ks:
        pos = K - 1
        # Compute m
        current_length = L
        m = 0
        while current_length < K:
            current_length *= 2
            m += 1
        # Process layers
        flip = 0
        layer = m
        current = current_length
        while layer > 0:
            prev_len = current // 2
            if pos >= prev_len:
                flip += 1
                pos -= prev_len
            current = prev_len
            layer -= 1
        # Determine character
        c = S[pos]
        if flip % 2 == 1:
            c = c.swapcase()
        ans.append(c)
    print(' '.join(ans))

if __name__ == '__main__':
    main()