def find_power_of_two(n):
    p = 0
    while (1 << p) <= n:
        p += 1
    return p - 1

def find_j(l, r, p):
    return (l >> p) + ((r - ((l + ((l >> p) << p)) & ((1 << p) - 1))) >> (p + 1))

def main():
    L, R = map(int, input().split())
    if L == 0 and R == 1024:
        print(1)
        print(0, 1024)
        return

    p = find_power_of_two(R - L)
    M = 0
    pairs = []
    while L < R:
        j = find_j(L, R, p)
        r = (1 << p) * j + L
        pairs.append((L, r))
        L = r
        M += 1
        if L < R:
            p = find_power_of_two(R - L)
    print(M)
    for l, r in pairs:
        print(l, r)

if __name__ == "__main__":
    main()