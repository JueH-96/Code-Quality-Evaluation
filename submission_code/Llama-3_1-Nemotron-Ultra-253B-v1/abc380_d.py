def main():
    import sys
    input = sys.stdin.read().split()
    S = input[0]
    Q = int(input[1])
    K_list = list(map(int, input[2:2+Q]))
    l = len(S)
    res = []
    for K in K_list:
        pos = K - 1
        original_pos = pos % l
        levels = pos // l
        flip = bin(levels).count('1') % 2
        c = S[original_pos]
        if flip:
            c = c.swapcase()
        res.append(c)
    print(' '.join(res))

if __name__ == "__main__":
    main()