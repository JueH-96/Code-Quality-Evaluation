import sys
import threading

def main():
    import sys

    data = sys.stdin
    line = data.readline().split()
    N = int(line[0]); M = int(line[1])
    pulls = []
    regs = []
    openers = []
    for _ in range(N):
        t, x = data.readline().split()
        t = int(t); x = int(x)
        if t == 0:
            pulls.append(x)
        elif t == 1:
            regs.append(x)
        else:
            openers.append(x)

    # Sort descending
    pulls.sort(reverse=True)
    regs.sort(reverse=True)
    openers.sort(reverse=True)

    Np = len(pulls)
    Nr = len(regs)
    E = len(openers)

    # Prefix sums
    P_prefix = [0] * (Np + 1)
    for i in range(Np):
        P_prefix[i+1] = P_prefix[i] + pulls[i]
    R_prefix = [0] * (Nr + 1)
    for i in range(Nr):
        R_prefix[i+1] = R_prefix[i] + regs[i]
    # Opener capacity prefix
    K_op = [0] * (E + 1)
    for i in range(E):
        K_op[i+1] = K_op[i] + openers[i]

    ans = 0
    # For bsearch
    for O in range(0, min(E, M) + 1):
        S = M - O  # slots for regs+pulls+zeros
        # total capacity to open regs
        K = K_op[O]
        # max regs that can give happiness
        Rk = K if K < Nr else Nr
        if S < 0:
            continue
        # If total candidate items (happiness>0) less than slots, we take all
        if Rk + Np <= S:
            # take all regs that can give happiness and all pulls
            val = R_prefix[Rk] + P_prefix[Np]
            if val > ans:
                ans = val
            continue

        # We need to choose b regs (from top Rk) and S-b pulls (from top Np)
        # b range is L..U
        L = S - Np
        if L < 0:
            L = 0
        U = Rk if Rk < S else S
        # binary search best b0 in [1..U] s.t. regs[b0-1] >= pull[S-b0]
        b0 = 0
        lo = 1
        hi = U
        # local references
        regs_arr = regs
        pulls_arr = pulls
        while lo <= hi:
            mid = (lo + hi) // 2
            # compare regs[mid-1] vs pull at index c = S-mid
            c = S - mid
            # pull_val = pulls_arr[c] if c < Np else 0
            if c < Np:
                pv = pulls_arr[c]
            else:
                pv = 0
            if regs_arr[mid-1] >= pv:
                b0 = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # clamp b0 into [L..U]
        if b0 < L:
            b = L
        elif b0 > U:
            b = U
        else:
            b = b0
        # compute sum
        # b regs from top, S-b pulls
        sum_regs = R_prefix[b]
        take_pulls = S - b
        if take_pulls < 0:
            take_pulls = 0
        if take_pulls > Np:
            take_pulls = Np
        sum_pulls = P_prefix[take_pulls]
        val = sum_regs + sum_pulls
        if val > ans:
            ans = val

    print(ans)

if __name__ == "__main__":
    main()