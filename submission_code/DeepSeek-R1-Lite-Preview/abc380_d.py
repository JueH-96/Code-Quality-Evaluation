def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    S = data[0]
    Q = int(data[1])
    Ks = list(map(int, data[2:2+Q]))
    
    L0 = len(S)
    res = []
    
    for K in Ks:
        # Find the smallest m such that L0 * 2^m >= K
        m = 0
        while (1 << m) * L0 < K:
            m += 1
        pos = K
        transformed = False
        current_m = m
        while current_m > 0:
            mid = L0 * (1 << (current_m - 1))
            if pos <= mid:
                current_m -= 1
            else:
                pos -= mid
                transformed = not transformed
                current_m -= 1
        char = S[pos - 1]
        if transformed:
            char = char.upper() if char.islower() else char.lower()
        res.append(char)
    print(' '.join(res))

if __name__ == '__main__':
    main()