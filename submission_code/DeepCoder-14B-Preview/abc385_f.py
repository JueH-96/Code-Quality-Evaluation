def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    buildings = []
    for _ in range(N):
        X = int(input[idx])
        H = int(input[idx + 1])
        buildings.append((X, H))
        idx += 2

    h_list = []
    for i in range(N):
        if i == 0:
            h_i = 0.0
        else:
            max_h = -float('inf')
            for j in range(i):
                Xj, Hj = buildings[j]
                Xi, Hi = buildings[i]
                numerator = Hj * Xi - Hi * Xj
                denominator = Xi - Xj
                if denominator == 0:
                    current = float('inf')
                else:
                    current = numerator / denominator
                if current > max_h:
                    max_h = current
            h_i = max_h if max_h != -float('inf') else 0.0
        h_list.append(h_i)

    H_max = max(h_list)
    if H_max > 0:
        print("{0:.15f}".format(H_max))
    else:
        found = False
        for i in range(1, N):
            if h_list[i] >= 0:
                found = True
                break
        if found:
            print("0.000000000000000000")
        else:
            print(-1)

if __name__ == '__main__':
    main()