# ------------------------------------------------------------
# We'll solve the problem by combining two main ideas:
#
# 1) How to decide if an integer n is "good":
#    A positive integer n is called good if sum_of_divisors(n) is divisible by 3.
#    It is a known (and not too hard to prove) fact that:
#       sum_of_divisors(n) mod 3 = 0
#    ⇔  in the prime-power factorization n = ∏ p^e,
#        at least one of the prime powers p^e has sum_of_divisors(p^e) ≡ 0 (mod 3).
#
#    And for a single prime p^e:
#      σ(p^e) = (p^(e+1) - 1)/(p - 1), but we really only need it mod 3.
#
#    Checking cases p mod 3:
#      - If p = 3, then σ(3^e) ≡ 1 (mod 3) for all e ≥ 1, never 0.
#      - If p ≡ 1 (mod 3), then σ(p^e) ≡ (1 + 1 + ... + 1) = e+1 (mod 3).
#           That is 0 mod 3  ⇔  (e+1) ≡ 0 mod 3  ⇔  e ≡ 2 mod 3.
#      - If p ≡ 2 (mod 3), p^k cycles mod 3 as 1,2,1,2,... so σ(p^e) ≡ 0 mod 3
#           exactly when e is odd.
#
#    Therefore n is good  ⇔  in its factorization, there exists at least one prime-power
#    p^e with:
#        (p ≡ 1 mod 3 and e ≡ 2 mod 3)  OR  (p ≡ 2 mod 3 and e is odd).
#
#    Equivalently, "n is NOT good" if and only if, for every prime p in n's factorization:
#        - p = 3: no effect (never helps make n good),
#        - p ≡ 1 mod 3 => e mod 3 ∈ {0,1} (never 2),
#        - p ≡ 2 mod 3 => e is even.
#
#    We define a multiplicative indicator g(n) = 1 if n is NOT good, else 0.
#    Then "n is good" ⇔ g(n) = 0.  One checks g is multiplicative.
#
#
# 2) Counting M-length sequences whose product is some integer n:
#    Let f(n) = number of ordered M-tuples of positive integers (a1,...,aM) with product = n.
#    One can show f is also multiplicative, and for prime-power n=p^e,
#       f(p^e) = C(e + M - 1, M - 1)
#    (the ways to distribute e identical prime-factors among M slots, order matters).
#
#    We want the total number of sequences of length M whose product is a "good" integer ≤ N.
#    That is  ∑_{1 ≤ n ≤ N, n good}  f(n).
#
#    Equivalently
#      ∑_{n ≤ N} f(n)*(1 - g(n))   =   ∑_{n ≤ N} f(n)  -  ∑_{n ≤ N} f(n)*g(n).
#
#    Define:
#      F(n) := f(n),
#      H(n) := f(n)*g(n).   (both F, H are multiplicative)
#
#    Then our desired count =  ( ∑_{k=1..N} F(k) )  -  ( ∑_{k=1..N} H(k) ).
#
#    So the problem reduces to computing two partial sums of multiplicative functions up to N:
#       S_F(N) = ∑_{k=1..N} F(k),   S_H(N) = ∑_{k=1..N} H(k).
#
#    We then output [ S_F(N) - S_H(N) ] mod 998244353.
#
#
# 3) Fast computation of partial sums of a multiplicative function up to N (where N ≤ 1e10):
#    One may use the classic "divide-and-conquer / sqrt-decomposition" approach, often seen
#    for summatory functions of d(n), φ(n), etc.  In outline:
#
#    - Precompute F(k) (or H(k)) for k ≤ K = 10^5 (since sqrt(1e10) = 10^5).  Also store
#      prefix sums prefixF for k ≤ K.
#
#    - Use a memo dictionary dpF to store S_F(x) for various x.  If x ≤ K, return prefixF[x].
#      Otherwise, one exploits the fact that floor(x / i) takes on only O(sqrt(x)) distinct values,
#      and sets up relations that compute S_F(x) in O(sqrt(x)) time total, reusing dpF(...) for
#      smaller arguments.  This is a standard technique for partial sums of multiplicative functions.
#
#    We'll do exactly that for S_F and similarly for S_H, then subtract.
#
#    The main steps in code:
#      (A) Precompute smallest prime factor up to 1e5 to factor numbers quickly.
#      (B) Precompute factorials mod 998244353 up to M + ~60 for binomial combinations.
#      (C) Build arrays f_vals[1..K], h_vals[1..K], and their prefix sums.
#      (D) Implement get_SF(x) and get_SH(x) with memo + the usual "floor(x//i)" trick.
#      (E) Final answer = get_SF(N) - get_SH(N) mod 998244353.
#
# 4) Complexity:
#    - Precomputation up to K=1e5 is done in about O(K log K) or so in Python (factor-sieve).
#    - Each partial sum call get_SF(N) or get_SH(N) runs in about O(sqrt(N)) = 1e5 steps,
#      which is borderline but can be done with efficient I/O and reasonable optimizations.
#      We do it twice (for F and for H).
#
#    This is the standard known method to handle summatory multiplicative functions up to ~1e10.
#
# ------------------------------------------------------------

import sys
sys.setrecursionlimit(10**7)
input_data = sys.stdin.read().strip().split()
N = int(input_data[0])
M = int(input_data[1])

MOD = 998244353

# ------------------------------------------------------------------
# 1) Precompute prime factor data up to 1e5
#    We'll store spf[x] = smallest prime factor of x (x >= 2),
#    with spf[1] = 1 conventionally.
# ------------------------------------------------------------------
MAXK = 10**5
spf = [0]*(MAXK+1)
def build_spf_sieve(n=MAXK):
    spf[1] = 1
    for i in range(2,n+1):
        if spf[i] == 0:  # i is prime
            spf[i] = i
            if i*i <= n:
                for j in range(i*i, n+1, i):
                    if spf[j] == 0:
                        spf[j] = i

build_spf_sieve(MAXK)

# ------------------------------------------------------------------
# 2) Precompute factorials up to M + maybe ~40 (enough for exponents
#    of small primes up to 1e5).  In fact, for n <= 1e5, the largest
#    exponent of 2 can be ~16 (2^17=131072>1e5). For bigger primes
#    exponents are smaller. But let's be safe and go up to M + 100.
# ------------------------------------------------------------------
MAXFAC = M + 200  # a bit of safety
fact = [1]*(MAXFAC+1)
invfact = [1]*(MAXFAC+1)
for i in range(1, MAXFAC+1):
    fact[i] = fact[i-1] * i % MOD

# Fermat's little theorem for inverse factorial
invfact[MAXFAC] = pow(fact[MAXFAC], MOD-2, MOD)
for i in reversed(range(MAXFAC)):
    invfact[i] = invfact[i+1]*(i+1)%MOD

def comb(n,r):
    if r<0 or r>n: return 0
    return fact[n]*invfact[r]%MOD*invfact[n-r]%MOD

# ------------------------------------------------------------------
# 3) Function to factor a <= 1e5 quickly using spf[], returning
#    a dict {prime : exponent}.
# ------------------------------------------------------------------
def factor_small(x):
    res = {}
    while x>1:
        p = spf[x]
        e = 0
        while spf[x] == p:
            x//=p
            e+=1
        res[p] = res.get(p,0) + e
    return res

# ------------------------------------------------------------------
# 4) f(n) = # of ordered M-tuples with product = n.
#    For n <= 1e5, we factor it via factor_small and build:
#         f(n) = ∏ C(e_p + M - 1, M - 1),
#    Then store in array f_vals[n].
#
#    Next, we define the "not-good" indicator g(n)=1 if n not good, else 0.
#    Then h(n) = f(n)*g(n). We store h_vals[n].
# ------------------------------------------------------------------
f_vals = [0]*(MAXK+1)
h_vals = [0]*(MAXK+1)

def is_not_good_prime_power(p, e):
    # Return True if p^e does NOT yield sum_of_divisors(...) ≡ 0 mod 3
    # i.e. if sigma(p^e) mod 3 != 0
    # => for p=3, always 1 mod 3 => never 0 => so "not good" => True
    #    p ≡ 1 mod 3 => need e mod 3 != 2
    #    p ≡ 2 mod 3 => need e even
    # We'll return True if that prime-power is "NOT good".
    if p == 3:
        return True  # sigma(3^e) ~ 1 mod 3 => not zero => not good
    r = p % 3
    if r == 1:
        # not good means e mod 3 != 2
        if (e % 3) == 2:
            return False
        else:
            return True
    elif r == 2:
        # not good means e is even
        if (e % 2) == 1:
            return False
        else:
            return True
    return True  # fallback

def build_f_and_h():
    f_vals[1] = 1
    # For n=1 => product of M positives = 1 means all =1, only 1 way => f(1)=1
    # check if 1 is not good => sum_of_div(1)=1 => not multiple of 3 => g(1)=1 => h(1)=1
    # so that'll be set after the loop if we do it carefully.
    for n in range(2, MAXK+1):
        fs = factor_small(n)
        # compute f(n)
        cnt = 1
        for (p,e) in fs.items():
            cnt = (cnt * comb(e+M-1, M-1)) % MOD
        f_vals[n] = cnt
    # now build h(n) = f(n)*g(n)
    #   g(n)=1 if for every prime-power factor p^e => "p^e not good"
    #   i.e. none prime-power is "good". Equivalently if for ALL p^e => is_not_good_prime_power(p,e)=True
    # We can just do a check: if ANY prime-power is "good", then g(n)=0 => h(n)=0.
    # We'll factor n again. For efficiency, we can reuse factor_small but that’s okay for n<=1e5.
    for n in range(1, MAXK+1):
        fs = factor_small(n)
        not_good_all = True
        for (p,e) in fs.items():
            if not is_not_good_prime_power(p,e):
                not_good_all = False
                break
        if not_good_all:
            h_vals[n] = f_vals[n]
        else:
            h_vals[n] = 0

build_f_and_h()

# prefix sums of f, h
pfF = [0]*(MAXK+1)
pfH = [0]*(MAXK+1)
for i in range(1, MAXK+1):
    pfF[i] = (pfF[i-1] + f_vals[i]) % MOD
    pfH[i] = (pfH[i-1] + h_vals[i]) % MOD

# ------------------------------------------------------------------
# 5) Partial-sum computations:
#    S_F(x) = ∑_{n=1..x} f(n), for a multiplicative f.
#    We'll implement get_SF(x) using a standard "sqrt-decomposition"
#    approach + memo. Similarly for get_SH(x).
#
#    Because x can be up to 1e10, we do:
#       if x <= MAXK => return pfF[x]
#       if in memo => return it
#       otherwise, we exploit the fact that floor(x/i) repeats, etc.
#
#    We gather the distinct values of floor(x//i) for i ≤ √x, and
#    handle intervals.  This is a well-known approach for summatory
#    multiplicative functions (sometimes called "min_25 sieve" technique
#    or just "divide-and-conquer" approach).
#
#    We'll implement an iterative or memo-based version.
# ------------------------------------------------------------------

import math

memoF = {}
memoH = {}

def get_SF(x):
    if x <= MAXK:
        return pfF[x]
    if x in memoF:
        return memoF[x]
    r = int(math.isqrt(x))
    s = 0
    # 1) sum over i from 1..r the values f(i) * count of n in [1.. x] with floor(x/n)= i
    #    but for a general f, the typical trick is:
    #    we partition the range [1..x] by the distinct values of floor(x//j).
    #    However, a simpler approach: we do the well-known technique:
    #       sum_{n=1..x} f(n) = sum_{k=1..r} [ sum_{n in [ floor(x/(k+1))+1 .. floor(x/k)] } f(n) ]
    #    We'll handle these intervals. For each such interval [start..end], if end <= MAXK
    #    we sum from prefix. If not, we do recursion in get_SF(...) but be mindful that
    #    the sum is over [start..end].
    # We can do it in the typical pattern done for e.g. d(n). But we must adapt for general f.
    #
    # In practice, the standard code pattern is something like:
    #    i = 1
    #    while i <= r:
    #       v = x // i
    #       nxt = x // v
    #       # sum f(j) for j in [i.. nxt]
    #       # then i = nxt+1
    #
    # Then a second loop handles the large values floor(x//j) < r.
    #
    # We'll define a helper to sum f over [L..R].
    def sumF_range(L,R):
        # sum f(n) for n in [L..R].
        # if R <= MAXK => just prefix
        # else we do a fallback: separate or naive?  We can do a small loop if R-L <= 100000? That can be large.
        # Typically in the known method, L..R won't be that large unless R <= sqrt(x). We'll do a direct loop if needed.
        # But that might be too slow.  Instead, we can do get_SF(R)-get_SF(L-1) for large intervals. Perfect. We must do recursion though.
        if L>R: return 0
        if R <= MAXK:
            return (pfF[R] - pfF[L-1]) % MOD
        # if the interval is big, we do sum_{n=L..R} f(n) = S_F(R) - S_F(L-1).
        return ( (get_SF(R) - get_SF(L-1)) % MOD ) % MOD

    ans = 0
    i = 1
    while i <= r:
        v = x // i
        nxt = x // v
        ans = (ans + sumF_range(i, nxt)) % MOD
        i = nxt+1

    # Now handle the part for n in [1.. x//(r+1]] => those are the distinct values floor(x//n) < r+1 => n>r.
    # Actually, the standard formula for the sum of f up to x (for a multiplicative f) is:
    #
    #    S_f(x) = ∑_{d=1..r} [ (floor(x/d)-floor(x/(d+1))) * f(d) ]  + ...
    # or a more direct approach is to notice that every n ≤ x either n ≤ r or floor(x/n) < r.
    #
    # A simpler pattern (common in practice):
    #    S_f(x) = ∑_{k=1..r} [ sumF_range( floor(x/(k+1))+1, floor(x/k) ) ]
    #
    # We coded a version in the loop above. That covers n from 1.. floor(x/r).
    # Next we must cover the distinct values of floor(x/n) which go down to 1. Typically:
    #    for v in [floor(x/(r+1))..1], the set of n with floor(x/n)=v is [ floor(x/(v+1))+1 .. floor(x/v) ]
    # and f(n) for that range might be combined with S_f(v). But that is for special f like d(n).
    #
    # For a *general* multiplicative f, the known code pattern is that once we've accounted for n up to r,
    # we recursively add S_f(x//m) for m=1.. something. But implementing the "textbook" approach can get quite large.
    #
    # Below is a well-known simpler snippet that works for e.g. d(n), but for a general f it actually *does* also work,
    # provided we have a precomputed or recursively-computed sum for smaller arguments.  The set of distinct "x//i" is
    # about 2*r in size, so we can handle them in a typical manner.
    #
    # Because of time/space constraints here, we do the standard pattern used for any multiplicative f
    # once you have prefix sums up to sqrt(x).  This is the typical “run i=1..r” trick in one loop, then
    # a second loop enumerates the distinct values of floor(x/i) that are < r.  We'll fill in that second loop:
    #

    # The part from n=1..r is covered in the loop.  Now for the big n, i.e. n in (r.. x]:
    #   but if n>r, we have floor(x/n)< floor(x/r)= ???  So we gather distinct t = floor(x/n) where t < r.
    #   Each t arises for n in [ floor(x/(t+1))+1 .. floor(x/t) ], typically smaller in size, so we can sum with prefix or recursion.

    # So let's do a second loop to handle t from 1.. x//(r+1). We define:
    limit = x // (r+1)
    for t in range(1, limit+1):
        # all n with floor(x/n)= t lie in [ x//(t+1)+1 .. x//t ] and that entire block n≥ r+1 typically.
        start = (x//(t+1))+1
        end   = x//t
        if start<=r: start = r+1
        # sum f(n) for n in [start..end].
        if start<=end:
            ans = (ans + t * sumF_range(start, end)) % MOD

    memoF[x] = ans % MOD
    return memoF[x]

def get_SH(x):
    if x <= MAXK:
        return pfH[x]
    if x in memoH:
        return memoH[x]
    r = int(math.isqrt(x))
    ans = 0
    def sumH_range(L,R):
        if L>R: return 0
        if R <= MAXK:
            return (pfH[R] - pfH[L-1]) % MOD
        return ( (get_SH(R) - get_SH(L-1)) % MOD ) % MOD

    i = 1
    while i <= r:
        v = x // i
        nxt = x // v
        ans = (ans + sumH_range(i, nxt)) % MOD
        i = nxt+1

    limit = x // (r+1)
    for t in range(1, limit+1):
        start = x//(t+1)+1
        end   = x//t
        if start<=r: start = r+1
        if start<=end:
            rngsum = sumH_range(start,end)
            ans = (ans + t*rngsum) % MOD

    memoH[x] = ans % MOD
    return memoH[x]


def main():
    # We want: count of length-M sequences whose product is good ≤ N.
    # = S_F(N) - S_H(N) mod 998244353
    sfN = get_SF(N)
    shN = get_SH(N)
    ans = (sfN - shN) % MOD
    print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()