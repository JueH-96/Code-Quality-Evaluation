import sys

def compute_prefix_sum(string, base, mod):
    N = len(string)
    prefix_sum = [0]
    current_sum = 0
    pow_val = 1  # base^0
    for i in range(N):
        char_val = (ord(string[i]) - ord('A') + 1) % mod
        term = (char_val * pow_val) % mod
        current_sum = (current_sum + term) % mod
        prefix_sum.append(current_sum)
        pow_val = (pow_val * base) % mod  # update for next power
    return prefix_sum

# Read input
data = sys.stdin.read().strip()
S = data
len_S = len(S)
REV = S[::-1]

# Define moduli and bases for double hashing
mod1 = 1000000007
base1 = 31
mod2 = 1000000009
base2 = 37

# Compute modular inverses for bases
inv_base1 = pow(base1, -1, mod1)
inv_base2 = pow(base2, -1, mod2)

# Compute prefix sums for S and REV with both hash functions
prefix_sum_S1 = compute_prefix_sum(S, base1, mod1)
prefix_sum_S2 = compute_prefix_sum(S, base2, mod2)
prefix_sum_REV1 = compute_prefix_sum(REV, base1, mod1)
prefix_sum_REV2 = compute_prefix_sum(REV, base2, mod2)

# Find the largest L where the suffix of length L is a palindrome
for L in range(len_S, 0, -1):
    # Check for first modulus
    sum_k_S1 = (prefix_sum_S1[len_S] - prefix_sum_S1[len_S - L] + mod1) % mod1
    hash_rel_S1 = (sum_k_S1 * pow(inv_base1, len_S - L, mod1)) % mod1
    hash_rel_REV1 = prefix_sum_REV1[L]
    equal1 = (hash_rel_S1 == hash_rel_REV1)
    
    # Check for second modulus
    sum_k_S2 = (prefix_sum_S2[len_S] - prefix_sum_S2[len_S - L] + mod2) % mod2
    hash_rel_S2 = (sum_k_S2 * pow(inv_base2, len_S - L, mod2)) % mod2
    hash_rel_REV2 = prefix_sum_REV2[L]
    equal2 = (hash_rel_S2 == hash_rel_REV2)
    
    if equal1 and equal2:
        # L is the length of the longest palindromic suffix
        break

# Compute the length of the prefix to reverse
len_prefix = len_S - L
# Reverse the prefix and append to S
reversed_part = S[:len_prefix][::-1]
result = S + reversed_part

# Output the result
print(result)