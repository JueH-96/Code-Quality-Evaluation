import sys

def read_sheet(h, w, it):
    s = [next(it).rstrip() for _ in range(h)]
    blacks = {(r, c) for r in range(h) for c in range(w) if s[r][c] == '#'}
    r_list = [r for r, _ in blacks]
    c_list = [c for _, c in blacks]
    return blacks, min(r_list), max(r_list), min(c_list), max(c_list)

def main() -> None:
    data = sys.stdin.read().strip().splitlines()
    it = iter(data)

    # Sheet A
    H_A, W_A = map(int, next(it).split())
    SA, rminA, rmaxA, cminA, cmaxA = read_sheet(H_A, W_A, it)

    # Sheet B
    H_B, W_B = map(int, next(it).split())
    SB, rminB, rmaxB, cminB, cmaxB = read_sheet(H_B, W_B, it)

    # Sheet X (target)
    H_X, W_X = map(int, next(it).split())
    SX, _, _, _, _ = read_sheet(H_X, W_X, it)

    # ranges of legal shifts for A and B
    def shift_range(r_min, r_max, upper):
        low = -r_min
        high = upper - 1 - r_max
        return range(low, high + 1) if low <= high else None

    r_range_A = shift_range(rminA, rmaxA, H_X)
    c_range_A = shift_range(cminA, cmaxA, W_X)
    r_range_B = shift_range(rminB, rmaxB, H_X)
    c_range_B = shift_range(cminB, cmaxB, W_X)

    # if a sheet already cannot be fully contained -> impossible
    if None in (r_range_A, c_range_A, r_range_B, c_range_B):
        print("No")
        return

    # brute force all pairs of shifts
    for drA in r_range_A:
        for dcA in c_range_A:
            shifted_A = {(r + drA, c + dcA) for r, c in SA}
            # little optimisation: if shifted A already has cells outside SX -> skip
            if not shifted_A.issubset(SX):
                # may still match after adding B, but if A alone already reaches a
                # position where X is '.', the union will never equal SX
                continue

            for drB in r_range_B:
                for dcB in c_range_B:
                    shifted_B = {(r + drB, c + dcB) for r, c in SB}
                    union = shifted_A | shifted_B
                    if union == SX:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()