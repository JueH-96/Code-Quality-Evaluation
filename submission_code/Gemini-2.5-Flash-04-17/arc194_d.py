import sys
from collections import Counter

# Set recursion depth, although not strictly necessary for this iterative solution
# sys.setrecursionlimit(3000) 

MOD = 998244353

# Precompute factorials and inverse factorials
# Max number of components k is N/2, max count of any component is k.
# Need fact[k] and inv_fact[c] for c <= k <= N/2.
# MAX_PRECOMP = N // 2 + 5 is sufficient for k and c values that appear in the formula.
# However, to be safe and simple, precompute up to N.
MAX_PRECOMP = 5005 
fact = [1] * MAX_PRECOMP
inv_fact = [1] * MAX_PRECOMP

for i in range(1, MAX_PRECOMP): # Start from 1 for fact calculation
    fact[i] = (fact[i-1] * i) % MOD

# Modular exponentiation (a^b % m)
def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

# Modular inverse using Fermat's Little Theorem (a^(MOD-2) % MOD)
def mod_inverse(n):
    return power(n, MOD - 2)

# Precompute inverse factorials
# inv_fact[i] = inv(i!)
inv_fact[MAX_PRECOMP - 1] = mod_inverse(fact[MAX_PRECOMP - 1])
for i in range(MAX_PRECOMP - 2, -1, -1): # Calculate down to 0
    inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD
# inv_fact[0] should be 1 (inv(0!) = inv(1) = 1), which is correctly initialized.

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    components = []
    balance = 0
    start_idx = 0
    for i in range(N):
        if S[i] == '(':
            balance += 1
        elif S[i] == ')':
            balance -= 1
        
        if balance == 0:
            # Found an irreducible component S[start_idx : i+1]
            components.append(S[start_idx : i+1])
            start_idx = i + 1

    # Count frequencies of components
    component_counts = Counter(components)

    # The number of distinct strings is k! / (c_1! * c_2! * ...)
    # where k is the total number of components and c_i is the count of the i-th distinct component

    k = len(components)
    
    # Numerator is k! mod MOD
    numerator = fact[k]

    # Denominator is product of c_i! mod MOD
    denominator = 1
    for count in component_counts.values():
        # Calculate product of inv_fact[count] % MOD
        # Since count <= k <= N/2 <= MAX_PRECOMP, inv_fact[count] is available.
        denominator = (denominator * inv_fact[count]) % MOD
    
    # Result is numerator * (denominator)^(-1) mod MOD
    ans = (numerator * denominator) % MOD
    print(ans)

# Main execution block
if __name__ == "__main__":
    solve()