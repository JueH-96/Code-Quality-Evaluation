import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    H = list(map(int, (next(it) for _ in range(n))))
    # Precompute prefix sums of damage for starting offsets s=0,1,2 (mod 3)
    # pref[s][r] = damage sum of first r actions when first action is at T mod3 == s
    pref = [[0]*4 for _ in range(3)]
    def damage(tmod3):
        return 3 if tmod3 == 0 else 1
    for s in range(3):
        for r in range(1, 4):
            tmod = (s + r - 1) % 3
            pref[s][r] = pref[s][r-1] + damage(tmod)

    total_T = 0
    # s = T1 mod 3 for the current enemy (first attack on this enemy)
    # initially T0 = 0 -> T1 = 1 -> s = 1 mod 3
    s = 1

    for h in H:
        full = h // 5
        rem = h - full * 5
        if rem == 0:
            r = 0
        else:
            # find minimal r in 1..3 with pref[s][r] >= rem
            # rem is in 1..4, and pref[s][3] == 5 always >= rem
            if pref[s][1] >= rem:
                r = 1
            elif pref[s][2] >= rem:
                r = 2
            else:
                r = 3
        k = full * 3 + r
        total_T += k
        # next starting offset
        s = (s + (k % 3)) % 3

    # Output result
    sys.stdout.write(str(total_T))

if __name__ == "__main__":
    main()