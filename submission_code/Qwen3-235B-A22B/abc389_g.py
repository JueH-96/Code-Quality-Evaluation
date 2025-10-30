import sys
from sys import stdin

def main():
    N, P = map(int, stdin.readline().split())

    max_fact = max(N-1, 30)
    fact = [1]*(max_fact+1)
    for i in range(1, max_fact+1):
        fact[i] = fact[i-1] * i % P

    inv_fact = [1]*(max_fact+1)
    inv_fact[max_fact] = pow(fact[max_fact], P-2, P)
    for i in range(max_fact-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % P

    def comb_mod(n, k):
        if k <0 or k >n:
            return 0
        return fact[n] * inv_fact[k] % P * inv_fact[n -k] % P

    def comb_bin(a, m):
        if m <0 or m >a:
            return 0
        num = 1
        denom = 1
        for i in range(m):
            num = num * (a -i) % P
            denom = denom * (i+1) % P
        return num * pow(denom, P-2, P) % P

    target = N // 2
    valid_sequences = []

    def dfs(current_seq, even_sum, odd_sum):
        sum_so_far = sum(current_seq)
        if sum_so_far == N:
            if even_sum == target and odd_sum == target:
                valid_sequences.append(current_seq.copy())
            return

        current_len = len(current_seq)
        next_parity = (current_len) % 2
        remaining = N - sum_so_far

        for s in range(1, remaining+1):
            new_even = even_sum
            new_odd = odd_sum
            if (next_parity == 0):
                new_even += s
            else:
                new_odd += s

            if new_even > target or new_odd > target:
                continue

            future_rem = remaining - s
            e_ok = (new_even <= target <= new_even + future_rem)
            o_ok = (new_odd <= target <= new_odd + future_rem)
            if not (e_ok and o_ok):
                continue

            current_seq.append(s)
            dfs(current_seq, new_even, new_odd)
            current_seq.pop()

    dfs([1], 1, 0)

    max_M = N * (N-1) // 2
    answer = [0] * (max_M + 1)

    def expand_binomial(a):
        poly = [0] * (a + 1)
        for k in range(a + 1):
            poly[k] = comb_bin(a, k)
        return poly

    def multiply(p1, p2, upto):
        res = [0] * (upto + 1)
        for i in range(len(p1)):
            if p1[i] == 0:
                continue
            for j in range(len(p2)):
                if i + j > upto:
                    continue
                res[i + j] = (res[i + j] + p1[i] * p2[j]) % P
        return res

    for S in valid_sequences:
        wtp = fact[N-1]
        for s in S[1:]:
            wtp = wtp * inv_fact[s] % P

        A = 0
        for s in S:
            A += s * (s - 1) // 2

        Q = [0] * (max_M + 1)
        for m in range(0, max_M + 1):
            Q[m] = comb_bin(A, m)

        GF = [0] * (max_M + 1)
        GF[0] = 1

        for d in range(1, len(S)):
            a = S[d-1]
            b = S[d]

            binom = expand_binomial(a)
            if binom:
                binom[0] = (binom[0] - 1) % P
            F = binom
            F_extended = [0] * (max_M + 1)
            for i in range(len(F)):
                if i <= max_M:
                    F_extended[i] = F[i]
            F = F_extended

            current = [0] * (max_M + 1)
            current[0] = 1
            exp = b
            power_poly = F[:]
            while exp > 0:
                if exp % 2 == 1:
                    current = multiply(current, power_poly, max_M)
                power_poly = multiply(power_poly, power_poly, max_M)
                exp //= 2

            GF = multiply(GF, current, max_M)

        GF = multiply(GF, Q, max_M)

        for m in range(max_M + 1):
            if GF[m]:
                contribution = (wtp * GF[m]) % P
                answer[m] = (answer[m] + contribution) % P

    output = []
    for m in range(N-1, max_M + 1):
        output.append(str(answer[m] % P))
    print(' '.join(output))

if __name__ == "__main__":
    main()