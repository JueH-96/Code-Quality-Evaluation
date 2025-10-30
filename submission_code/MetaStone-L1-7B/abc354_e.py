def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    cards = []
    for _ in range(n):
        a = int(input[idx])
        b = int(input[idx+1])
        cards.append((a, b))
        idx += 2

    moves = []
    for i in range(n):
        for j in range(i + 1, n):
            a_i, b_i = cards[i]
            a_j, b_j = cards[j]
            if a_i == a_j or b_i == b_j:
                moves.append((i, j))

    max_mask = 1 << n
    g = [0] * max_mask

    # Sort masks by the number of set bits
    masks = sorted(range(max_mask), key=lambda x: bin(x).count('1'))

    for mask in masks:
        reachable = set()
        for (i, j) in moves:
            if (mask & (1 << i)) and (mask & (1 << j)):
                new_mask = mask ^ (1 << i) ^ (1 << j)
                reachable.add(g[new_mask])
        mex = 0
        while mex in reachable:
            mex += 1
        g[mask] = mex

    full_mask = (1 << n) - 1
    if g[full_mask] != 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()