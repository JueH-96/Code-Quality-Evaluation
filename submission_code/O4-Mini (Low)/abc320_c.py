def main():
    import sys
    M = int(sys.stdin.readline().strip())
    S = [sys.stdin.readline().strip() for _ in range(3)]
    # positions[digit][i] = list of p in 0..M-1 where S[i][p]==digit
    positions = {str(d): [[], [], []] for d in range(10)}
    for i in range(3):
        for p, ch in enumerate(S[i]):
            positions[ch][i].append(p)
    best = float('inf')
    # try each digit
    for d in positions:
        P = positions[d]
        # need non-empty for all 3 reels
        if not P[0] or not P[1] or not P[2]:
            continue
        # permutations of reel indices
        from itertools import permutations
        for perm in permutations((0,1,2), 3):
            i, j, k = perm
            Pi, Pj, Pk = P[i], P[j], P[k]
            # for each choice of base positions
            for pi in Pi:
                t_i = pi
                # if t_i already >= best, no need
                if t_i >= best:
                    continue
                # try each p_j
                for pj in Pj:
                    # compute minimal t_j > t_i with t_j ≡ pj (mod M)
                    if pj > t_i:
                        t_j = pj
                    else:
                        # need ceil((t_i+1 - pj)/M)
                        delta = t_i + 1 - pj
                        k_mul = (delta + M - 1) // M
                        t_j = pj + k_mul * M
                    if t_j >= best:
                        continue
                    # now p_k
                    for pk in Pk:
                        if pk > t_j:
                            t_k = pk
                        else:
                            delta2 = t_j + 1 - pk
                            k2 = (delta2 + M - 1) // M
                            t_k = pk + k2 * M
                        # t_i < t_j < t_k by construction
                        # T is t_k
                        if t_k < best:
                            best = t_k
    if best == float('inf'):
        print(-1)
    else:
        print(best)

if __name__ == "__main__":
    main()