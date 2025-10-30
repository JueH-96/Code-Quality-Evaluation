import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    H = list(map(int, data[1:]))

    T = 0  # total time / number of attacks so far

    for h in H:
        # We need the smallest k >= 0 such that
        # if we start at time T (so next attack is T+1),
        # the sum of damages over k attacks is >= h.
        # Damage pattern repeats every 3 attacks: [1,1,3],
        # but offset by (T mod 3).

        p = T % 3  # offset: at i-th attack (1-based), we look at (p + i) % 3
        # target residue r for i mod 3 that gives a 3-damage hit:
        r = (-p) % 3  # we want (p+i) % 3 == 0 => i % 3 == -p mod 3

        # Define function to compute total damage after k attacks
        def damage_after(k):
            # count of heavy hits (3 damage) = number of i in [1..k] with i mod 3 == r
            if r == 0:
                c3 = k // 3
            else:
                if k < r:
                    c3 = 0
                else:
                    c3 = (k - r) // 3 + 1
            # total damage = c3*3 + (k - c3)*1 = k + 2*c3
            return k + 2 * c3

        # binary search minimal k such that damage_after(k) >= h
        lo, hi = 0, (h // 3 + 2) * 3  # an upper bound on k: at most 3*(h/3+2)
        while lo < hi:
            mid = (lo + hi) // 2
            if damage_after(mid) >= h:
                hi = mid
            else:
                lo = mid + 1

        k = lo
        T += k

    print(T)

if __name__ == "__main__":
    main()