import sys
import threading

def main():
    data = sys.stdin.read().split()
    S = data[0]
    Q = int(data[1])
    Ks = list(map(int, data[2:]))

    L = len(S)
    # Find the total length after enough doublings to cover the max K
    maxK = max(Ks)
    total_len = L
    # Double until total_len >= maxK
    while total_len < maxK:
        total_len <<= 1

    def invert_char(c):
        # Swap case
        if 'a' <= c <= 'z':
            return chr(ord(c) - ord('a') + ord('A'))
        else:
            return chr(ord(c) - ord('A') + ord('a'))

    res = []
    for K in Ks:
        k = K
        flip = False
        curr_len = total_len
        # Walk down the doubling tree until we reach the original string size
        while curr_len > L:
            half = curr_len >> 1
            if k > half:
                # it's in the second half -> corresponds to inverted first half
                k -= half
                flip = not flip
            # either way we go into the first-half block of the previous iteration
            curr_len = half
        # Now curr_len == L, so k is in [1..L]
        ch = S[k-1]
        if flip:
            ch = invert_char(ch)
        res.append(ch)

    # Output answers separated by spaces
    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()