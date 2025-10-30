#!/usr/bin/env python3
MOD = 998244353

# Pre-calculate primes up to 1000.
def sieve(n):
    isprime = [True]*(n+1)
    isprime[0]=isprime[1]=False
    primes = []
    for i in range(2,n+1):
        if isprime[i]:
            primes.append(i)
            for j in range(i*i, n+1, i):
                isprime[j] = False
    return primes

# Given an integer x, returns the exponent of prime p in x.
def v_p(x, p):
    cnt = 0
    while x % p == 0:
        x //= p
        cnt += 1
    return cnt

# For a given prime p and its sequence b[1..L] (with L = N-1) where:
#   b[1] = v_p(A1) and for 2<=i<=L, b[i] = v_p(Ai)
# perform DP on a “walk” of N nodes (N = L+1) 
# where we must choose the base (2 ways if b[1]>0, one way if b[1]==0)
# and for i=2..N-1 use the recurrence:
#    d[i+1] = d[i] ± b[i]  (if b[i]>0; if b[i]==0, then only one possibility d[i+1]=d[i])
# In addition, whenever a new node with exponent d' is produced,
# we multiply our weight by p^(d'). (The weight accumulates the sum of exponents.)
# (Once a zero appears in the walk, the overall min=0 condition is satisfied.)
def dp_for_prime(b, p, N):
    # dp is a list of length (max_possible_exponent+1) 
    # dp[d] = total weight for sequences that have current exponent d.
    # The maximum possible exponent never exceeds (b[1] + sum(b[2..N-1])).
    max_possible = (b[0] if b else 0) + sum(b[1:])  # here b[0] is for A1.
    dp = [0]*(max_possible+1)
    # Base: for node1 and node2.
    # For A1 = b[0]:
    if b and b[0] > 0:
        # Two distinct choices.
        # option X: d1 = b[0], d2 = 0, total contribution = p^(b[0] + 0)
        val = pow(p, b[0], MOD)
        dp[0] = (dp[0] + val) % MOD
        # option Y: d1 = 0, d2 = b[0], weight = p^(0 + b[0])
        dp[b[0]] = (dp[b[0]] + val) % MOD
    else:
        # When A1==1 so b[0]==0, there's only one possibility: d1=d2=0.
        dp[0] = 1
    # Now there are N-2 transitions from node2 to nodeN.
    # We have a sequence b[1], b[2], ... b[N-1] with indices 1..(N-1) for A2...A_(N-1).
    # There are total N nodes.
    # Our current dp corresponds to node2.
    for i in range(1, N-1):   # i from 1 to N-2; transition using b[i]
        a = b[i]  # a = b[i] for edge A_{i+1}
        new_max = max_possible  # same bound
        newdp = [0]*(new_max+1)
        for d, ways in enumerate(dp):
            if ways == 0:
                continue
            # next exponent d_next will be determined by the choice.
            if a == 0:
                # Only one possibility: d_next = d.
                # Multiply by p^(d) (because we add the new node’s exponent)
                newdp[d] = (newdp[d] + ways * pow(p, d, MOD)) % MOD
            else:
                # Option Y: add
                nd = d + a
                if nd <= new_max:
                    newdp[nd] = (newdp[nd] + ways * pow(p, nd, MOD)) % MOD
                # Option X: subtract, possible only if d >= a.
                if d >= a:
                    nd = d - a
                    newdp[nd] = (newdp[nd] + ways * pow(p, nd, MOD)) % MOD
        dp = newdp
    # At the end, dp corresponds to node N.
    # The gcd-condition (min exponent = 0) is automatically met since in the base at least one node was 0.
    return sum(dp) % MOD

def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N-1)]
    
    # Our answer will be computed prime-by-prime.
    # For each prime that appears in some A, build the list b:
    # b[0] = v_p(A[0]) and for i>=1: b[i] = v_p(A[i]).
    pr = sieve(1000)
    ans = 1
    for p in pr:
        # Check if p divides at least one A.
        flag = False
        for x in A:
            if x % p == 0:
                flag = True
                break
        if not flag:
            continue
        # Build b of length N-1.
        b = [ v_p(x, p) for x in A ]
        # Run DP for this prime. (There are N nodes.)
        Fp = dp_for_prime(b, p, N)
        ans = (ans * Fp) % MOD
    print(ans)
    
if __name__ == '__main__':
    main()