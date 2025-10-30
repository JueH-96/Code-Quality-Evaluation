# YOUR CODE HERE
MOD = 998244353

def compute_v_n(n):
    # Calculate the length of n when represented as a string
    length = len(str(n))
    # Calculate 10^(length * n) - 1
    power = pow(10, length * n, MOD * (10**length - 1))
    # Since 10^(length * n) mod (MOD * (10^length - 1)) is not directly useful, we need to find a better way
    # Instead, we can compute (10^(length * n) - 1) / (10^length - 1) * n mod MOD
    # But since 10^length - 1 is not necessarily coprime with MOD, we need to handle it differently
    # Alternatively, we can compute the geometric series sum:
    # V_N = n * (10^{0} + 10^{length} + 10^{2*length} + ... + 10^{(n-1)*length})
    # The sum inside is a geometric series with ratio 10^length, and it sums to (10^{length * n} - 1) / (10^length - 1)
    # So V_N = n * (10^{length * n} - 1) / (10^length - 1)
    # To compute this modulo MOD, we need to compute the modular inverse of (10^length - 1) modulo MOD
    # But since MOD is prime and 10^length - 1 is not divisible by MOD, the inverse exists
    # Compute 10^length mod MOD
    ten_length = pow(10, length, MOD)
    # Compute (10^length - 1) mod MOD
    denominator = (ten_length - 1) % MOD
    # Compute the modular inverse of denominator
    # Since MOD is prime, we can use Fermat's little theorem: inv = pow(denominator, MOD-2, MOD)
    inv_denominator = pow(denominator, MOD-2, MOD)
    # Compute 10^{length * n} mod (MOD * (10^length - 1)) is not straightforward, so we need to find a better way
    # Instead, compute 10^{length * n} mod (MOD * (10^length - 1)) is not necessary
    # We can compute 10^{length * n} mod MOD and 10^{length * n} mod (10^length - 1) separately and use the Chinese Remainder Theorem
    # But it's easier to compute the entire expression modulo MOD
    # So V_N mod MOD = n * (10^{length * n} - 1) / (10^length - 1) mod MOD
    # Which is equivalent to n * (10^{length * n} - 1) * inv_denominator mod MOD
    # Compute 10^{length * n} mod MOD
    ten_power = pow(10, length * n, MOD)
    numerator = (ten_power - 1) % MOD
    # Compute V_N mod MOD
    v_n_mod = (n * numerator * inv_denominator) % MOD
    return v_n_mod

n = int(input())
print(compute_v_n(n))