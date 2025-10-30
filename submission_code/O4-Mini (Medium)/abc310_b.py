def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    prices = []
    funcs = []
    for _ in range(N):
        data = list(map(int, input().split()))
        p = data[0]
        c = data[1]
        f = set(data[2:])
        prices.append(p)
        funcs.append(f)

    # Check all ordered pairs (i, j), i != j
    # We want j to be strictly superior to i:
    # 1) prices[j] <= prices[i]
    # 2) funcs[i] is subset of funcs[j]
    # 3) Either prices[j] < prices[i] OR funcs[j] strictly bigger than funcs[i]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if prices[j] <= prices[i] and funcs[i].issubset(funcs[j]):
                # check strictness
                if prices[j] < prices[i] or len(funcs[j]) > len(funcs[i]):
                    print("Yes")
                    return

    print("No")


if __name__ == "__main__":
    main()