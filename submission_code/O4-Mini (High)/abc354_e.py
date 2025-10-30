import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    A = [0]*N
    B = [0]*N
    for i in range(N):
        a,b = map(int, input().split())
        A[i], B[i] = a, b

    # Precompute all removable pairs as bitmasks
    pairs = []
    for i in range(N):
        ai, bi = A[i], B[i]
        for j in range(i+1, N):
            if ai == A[j] or bi == B[j]:
                pairs.append((1<<i) | (1<<j))

    max_mask = 1<<N
    # win[mask] = 1 if the position 'mask' is winning, 0 otherwise
    win = bytearray(max_mask)

    # DP over all masks in increasing numeric order.
    # For each mask, try removing any valid pair. If you can move
    # to a losing position, then mask is winning.
    for mask in range(max_mask):
        for pm in pairs:
            # check if both cards in pair pm are present in mask
            if (mask & pm) == pm:
                # next position after removing pm
                nxt = mask ^ pm
                # if that position is losing, current is winning
                if win[nxt] == 0:
                    win[mask] = 1
                    break

    # full mask: all cards present
    full = max_mask - 1
    print("Takahashi" if win[full] else "Aoki")

if __name__ == "__main__":
    main()