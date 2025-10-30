def minimal_good_sequences(L, R):
    blocks = []
    current = L
    while current < R:
        max_k = 0
        for k in range(60, -1, -1):
            block_size = 1 << k
            if current + block_size > R:
                continue
            if current % block_size == 0:
                max_k = k
                break
        if max_k == 0:
            blocks.append((current, R - 1))
            current = R
        else:
            block_size = 1 << max_k
            end = current + block_size
            blocks.append((current, end - 1))
            current = end
    return blocks

def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    blocks = minimal_good_sequences(L, R)
    print(len(blocks))
    for (l, r) in blocks:
        print(l, r)

if __name__ == "__main__":
    main()