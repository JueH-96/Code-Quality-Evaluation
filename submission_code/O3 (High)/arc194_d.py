import sys

MOD = 998244353

def fact_and_inv(up_to):
    fact = [1]*(up_to+1)
    for i in range(1, up_to+1):
        fact[i] = fact[i-1]*i % MOD
    inv_fact = [1]*(up_to+1)
    inv_fact[up_to] = pow(fact[up_to], MOD-2, MOD)
    for i in range(up_to, 0, -1):
        inv_fact[i-1] = inv_fact[i]*i % MOD
    return fact, inv_fact

def main() -> None:
    N   = int(sys.stdin.readline())
    S   = sys.stdin.readline().strip()

    # 1. split into primitive Dyck words
    pieces = []
    bal, start = 0, 0
    for i, ch in enumerate(S):
        bal += 1 if ch == '(' else -1
        if bal == 0:
            pieces.append(S[start:i+1])
            start = i+1
    k = len(pieces)

    # 2. multiplicities of equal primitives
    from collections import Counter
    cnt = Counter(pieces)

    # 3. factorials
    fact, inv_fact = fact_and_inv(k)

    # 4. multinomial
    ans = fact[k]
    for c in cnt.values():
        ans = ans * inv_fact[c] % MOD

    print(ans)

if __name__ == "__main__":
    main()