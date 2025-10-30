import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # Precompute parameters
    # We'll only search j from 0..Jsmall where Jsmall = floor(log2(N)) + 2
    # This keeps candidate count ~O(N·log N) and total work ~O(N^2 log N) ~ 14e6 for N=1000
    Nbit = N.bit_length()
    jsmall = Nbit + 2
    f_max = 0
    visited = set()
    bl = int.bit_length  # alias for speed
    A_list = A
    # For each j, group A_k by residue mod 2^(j+2), form candidates,
    # then evaluate f(i) = sum_k M(i + A_k) using the paper‐folding formula
    for j in range(jsmall + 1):
        mod = 1 << (j + 2)
        mask = mod - 1
        base = 3 << j  # 3 * 2^j
        # group counts by r = A_k mod mod
        group = {}
        for ak in A_list:
            r = ak & mask
            group[r] = group.get(r, 0) + 1
        # for each residue class produce candidate i (canonical in [1..mod])
        for r, cnt in group.items():
            # i ≡ (3*2^j - r) mod 2^(j+2)
            i0 = (base - r) & mask
            # choose representative >=1
            if i0:
                i = i0
            else:
                # if i0 == 0, choose i = mod (same class mod)
                i = mod
            if i in visited:
                continue
            visited.add(i)
            # evaluate f(i)
            fcount = 0
            # we can break early if even all remaining can't beat f_max
            for idx, ak in enumerate(A_list):
                n = i + ak
                # compute t = 2^{v2(n)}, trailing-zero power
                t = n & -n
                tz = bl(t) - 1
                # m = n >> tz; check m mod 4 == 3
                if ((n >> tz) & 3) == 3:
                    fcount += 1
                # early exit if impossible to exceed current best
                # even if all remaining positions were 1, total <= f_max
                if fcount + (N - idx - 1) <= f_max:
                    break
            if fcount > f_max:
                f_max = fcount
    # Print the maximum found
    sys.stdout.write(str(f_max))

if __name__ == "__main__":
    main()