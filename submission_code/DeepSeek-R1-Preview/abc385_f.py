def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    if N <= 1:
        print(-1)
        return
    buildings = []
    for _ in range(N):
        X = int(input[idx])
        H = int(input[idx + 1])
        buildings.append((X, H))
        idx += 2
    H_max = -float('inf')
    for i in range(1, N):
        Xj, Hj = buildings[i-1]
        Xi, Hi = buildings[i]
        numerator = Hj * Xi - Hi * Xj
        denominator = Xi - Xj
        if denominator == 0:
            continue
        C_ij = numerator / denominator
        if C_ij > H_max:
            H_max = C_ij
    if H_max < 0:
        print(-1)
    else:
        print("{0:.15f}".format(H_max))
        
if __name__ == '__main__':
    main()