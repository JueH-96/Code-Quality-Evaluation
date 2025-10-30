import sys
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # Prepare grouping by exponent of 2 and odd part
    maxA = 0
    A = [0]*N
    for i in range(N):
        ai = int(next(it))
        A[i] = ai
        if ai > maxA:
            maxA = ai
    # Maximum possible exponent (v2) is bit_length of maxA
    E = maxA.bit_length() + 1
    C_e = [0] * E
    S_e = [0] * E
    b_lists = [[] for _ in range(E)]
    # Group A_i by their 2-adic exponent and store odd part
    for ai in A:
        # trailing zeros count
        tz = (ai & -ai).bit_length() - 1
        bi = ai >> tz
        C_e[tz] += 1
        S_e[tz] += bi
        b_lists[tz].append(bi)
    # Part 1: pairs with e_i < e_j
    ans1 = 0
    max_e = 0
    for e in range(E):
        if C_e[e] > 0:
            max_e = e
    for e1 in range(max_e + 1):
        c1 = C_e[e1]
        if c1 == 0:
            continue
        s1 = S_e[e1]
        for e2 in range(e1 + 1, max_e + 1):
            c2 = C_e[e2]
            if c2 == 0:
                continue
            s2 = S_e[e2]
            # sum of f for all pairs in bucket e1 and bucket e2
            # f = B_i + B_j * 2^{e2-e1}
            ans1 += s1 * c2 + c1 * (s2 << (e2 - e1))
    # Part 2: i == j terms, f(2*A_i) = odd part of 2^{e+1} * B_i = B_i
    ans2 = sum(S_e)
    # Part 3: within each bucket e_i = e_j
    # We need sum_{i<j} f(B_i + B_j)
    # We'll compute by counting sums divisible by powers of two
    threshold = 1 << 17  # up to 131072 use list, else dict
    ans3 = 0
    for e in range(max_e + 1):
        b_list = b_lists[e]
        n = len(b_list)
        if n <= 1:
            continue
        S_group = S_e[e]
        # T1 = sum_{i<j} (B_i + B_j)
        # = S_group * (n-1)
        T1 = S_group * (n - 1)
        # initial sum3 = T1/2
        sum3 = T1 >> 1
        # determine highest k where 2^k <= 2 * max B
        maxB = max(b_list)
        max_sum = maxB * 2
        max_k = max_sum.bit_length() - 1
        # for each k>=2 subtract T_k / 2^k
        for k in range(2, max_k + 1):
            M = 1 << k
            mask = M - 1
            # compute T_k = sum_{i<j, (B_i+B_j)%M==0} (B_i+B_j)
            if M <= threshold:
                cnt = [0] * M
                sum_res = [0] * M
                for B in b_list:
                    r = B & mask
                    cnt[r] += 1
                    sum_res[r] += B
                T_k = 0
                cnt_r = cnt
                sum_r = sum_res
                for r in range(M):
                    c = cnt_r[r]
                    if not c:
                        continue
                    comp = (M - r) & mask
                    if r < comp:
                        c2 = cnt_r[comp]
                        if c2:
                            T_k += sum_r[r] * c2 + sum_r[comp] * c
                    elif r == comp:
                        # pairs within same residue
                        if c > 1:
                            T_k += sum_r[r] * (c - 1)
                # done list method
            else:
                cnt = {}
                sum_res = {}
                for B in b_list:
                    r = B & mask
                    cnt[r] = cnt.get(r, 0) + 1
                    sum_res[r] = sum_res.get(r, 0) + B
                T_k = 0
                processed = set()
                for r, c in cnt.items():
                    if r in processed:
                        continue
                    comp = (M - r) & mask
                    if comp == r:
                        # pairs inside same residue
                        if c > 1:
                            T_k += sum_res[r] * (c - 1)
                        processed.add(r)
                    else:
                        c2 = cnt.get(comp, 0)
                        if c2:
                            T_k += sum_res[r] * c2 + sum_res[comp] * c
                        processed.add(r)
                        processed.add(comp)
                # done dict method
            # subtract contribution
            sum3 -= (T_k >> k)
        ans3 += sum3
    # total answer
    ans = ans1 + ans2 + ans3
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()