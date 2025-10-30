#!/usr/bin/env python3
import sys
from functools import cmp_to_key

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    K = int(next(it))
    # Group by (A,B)
    cnt = {}
    for _ in range(N):
        A = int(next(it)); B = int(next(it))
        cnt[(A,B)] = cnt.get((A,B), 0) + 1
    # Build list of types (A,B,count)
    types = [(A, B, c) for (A,B), c in cnt.items()]
    # Comparator to sort by ratio r = B/(A-1), with A==1 treated as +inf
    def cmp_ratio(t1, t2):
        A1, B1, _ = t1
        A2, B2, _ = t2
        # treat A==1 as infinite ratio
        if A1 == 1 and A2 == 1:
            return 0
        if A1 == 1:
            return 1
        if A2 == 1:
            return -1
        # compare B1/(A1-1) vs B2/(A2-1) by cross-multiplying
        # B1*(A2-1) ? B2*(A1-1)
        v = B1*(A2-1) - B2*(A1-1)
        if v < 0:
            return -1
        if v > 0:
            return 1
        return 0

    types.sort(key=cmp_to_key(cmp_ratio))

    # DP[t] = list of undominated states (alpha, beta) for using t functions
    DP = [[] for _ in range(K+1)]
    DP[0] = [(1, 0)]

    # Process each type in increasing ratio order
    for (A, B, c) in types:
        # We can take up to l_max copies of this type (cannot exceed K)
        l_max = c if c < K else K

        # Precompute A^l and S[l] = B*(A^l -1)/(A-1) or B*l if A==1
        powA = [1] * (l_max + 1)
        S    = [0] * (l_max + 1)
        if A == 1:
            # f(x)=x+B, composition l times: f^l(x)=x + l*B
            for l in range(1, l_max+1):
                powA[l] = 1
                S[l] = B * l
        else:
            pa = 1
            for l in range(1, l_max+1):
                pa *= A
                powA[l] = pa
                # S[l] = B * (A^l - 1) // (A - 1)
                # integer division exact since (A^l -1) divisible by (A-1)
                S[l] = B * (pa - 1) // (A - 1)

        # Snapshot DP_old so we only extend from before this type
        DP_old = [DP[t][:] for t in range(K+1)]

        # Try picking l copies of this type on top of t_cur picks
        for t_cur in range(K+1):
            base_states = DP_old[t_cur]
            if not base_states:
                continue
            # largest l so that t_cur + l <= K
            max_l = l_max if t_cur + l_max <= K else (K - t_cur)
            # combine
            for (alpha, beta) in base_states:
                # for each possible number of copies l
                # new state uses l duplicates innermost of this type
                for l in range(1, max_l + 1):
                    t_new = t_cur + l
                    na = alpha * powA[l]
                    nb = alpha * S[l] + beta
                    # insert (na,nb) into DP[t_new] with Pareto prune
                    lst = DP[t_new]
                    # check if already dominated
                    dominated = False
                    for (ea, eb) in lst:
                        if ea >= na and eb >= nb:
                            dominated = True
                            break
                    if dominated:
                        continue
                    # otherwise remove any states that this one dominates
                    new_lst = []
                    for (ea, eb) in lst:
                        if not (na >= ea and nb >= eb):
                            new_lst.append((ea, eb))
                    new_lst.append((na, nb))
                    DP[t_new] = new_lst

    # Final answer: maximum alpha + beta in DP[K]
    ans = 0
    for (a, b) in DP[K]:
        v = a + b
        if v > ans:
            ans = v
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()