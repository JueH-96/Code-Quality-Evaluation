import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    it = iter(sys.stdin.buffer.read().split())
    n = int(next(it))

    t = [0] * n
    x = [0] * n

    # index (in the order they appear) of each potion event
    potion_idx = [-1] * n
    potion_cnt = 0
    for i in range(n):
        ti = int(next(it))
        xi = int(next(it))
        t[i] = ti
        x[i] = xi
        if ti == 1:                # potion
            potion_idx[i] = potion_cnt
            potion_cnt += 1

    # answer for every potion event (0 = discard, 1 = pick)
    pick = [0] * potion_cnt

    need = [0] * (n + 2)          # need[type] : monsters of that type still unmatched
    total_need = 0                # total monsters still unmatched
    k_min = 0                     # minimal possible maximal inventory size

    # scan from the end to the beginning
    for i in range(n - 1, -1, -1):
        if t[i] == 2:                           # monster
            need[x[i]] += 1
            total_need += 1
        else:                                   # potion
            idx = potion_idx[i]
            if need[x[i]]:                      # potion is necessary
                need[x[i]] -= 1
                total_need -= 1
                pick[idx] = 1                   # take it
            # otherwise we can safely discard it (pick[idx] stays 0)
        if total_need > k_min:
            k_min = total_need

    # if some monsters are still unmatched, it is impossible
    if total_need:
        print(-1)
        return

    print(k_min)
    if potion_cnt:
        print(' '.join(map(str, pick)))


if __name__ == "__main__":
    main()