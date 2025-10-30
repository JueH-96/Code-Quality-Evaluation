def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    pairs = []
    for _ in range(N):
        L = int(input[ptr])
        R = int(input[ptr + 1])
        pairs.append((L, R))
        ptr += 2
    L_sum = sum(L for L, R in pairs)
    if L_sum > 0:
        print("No")
        return
    D = -L_sum
    total_available = sum(R - L for L, R in pairs)
    if D > total_available:
        print("No")
        return
    X = [L for L, R in pairs]
    remaining = D
    for i in range(N):
        if remaining <= 0:
            break
        adjust = min(remaining, pairs[i][1] - pairs[i][0])
        X[i] += adjust
        remaining -= adjust
    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()