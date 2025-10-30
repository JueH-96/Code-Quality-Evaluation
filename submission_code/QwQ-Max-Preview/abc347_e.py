def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    x_list = list(map(int, input[ptr:ptr+Q]))
    ptr += Q

    current_s = 0
    S = set()
    s = []
    for xi in x_list:
        if xi in S:
            S.remove(xi)
            current_s -= 1
        else:
            S.add(xi)
            current_s += 1
        s.append(current_s)
    
    # Compute prefix sum
    prefix = [0] * (Q + 1)
    for i in range(Q):
        prefix[i + 1] = prefix[i] + s[i]
    
    # Prepare toggle positions for each j
    toggles = defaultdict(list)
    for idx, xi in enumerate(x_list):
        toggles[xi].append(idx)
    
    ans = [0] * N
    for j in range(1, N + 1):
        ts = toggles.get(j, [])
        total = 0
        m = len(ts)
        for i in range(m):
            if i % 2 == 0:
                start = ts[i]
                if i + 1 < m:
                    end = ts[i + 1]
                else:
                    end = Q
                total += prefix[end] - prefix[start]
        ans[j - 1] = total
    
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()