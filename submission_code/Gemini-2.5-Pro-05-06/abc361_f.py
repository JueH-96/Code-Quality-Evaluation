import math

N = int(input())

# C1: Count of perfect squares k^2 <= N (k >= 1). This is floor(sqrt(N)).
# This includes 1 = 1^2.
# math.isqrt(N) correctly computes floor(sqrt(N)).
count_C1 = math.isqrt(N)

# set_C2_candidates: Stores numbers a^b where a >= 2, b >= 3, and a^b <= N.
# Using a set handles duplicates (e.g., 2^6 = 64, which is also 4^3).
set_C2_candidates = set()

# Iterate 'a' from 2.
# The loop for 'a' can stop when a^3 > N.
a = 2
while True:
    # Calculate a^2 first to check if a^3 would overflow N using a more robust check.
    # a_aa = a * a
    # if a_aa > N // a:  This is equivalent to a^3 > N.
    #    break
    # current_power_val = a_aa * a
    # The above is slightly more complex; direct computation of a^3 is fine in Python.
    # Python's integers have arbitrary precision.
    
    # Check if a itself has grown too large.
    # Max 'a' is approx N^(1/3). For N=10^18, this is 10^6.
    # If 'a' exceeds this by a small margin, a^3 will surely exceed N.
    # This check helps avoid large 'a' values if N is small.
    if a > 10**6 + 5 : # Optimization: (10^6)^3 = 10^18. If N <= 10^18, 'a' won't need to be much larger than 10^6.
        break

    a_cubed = a * a * a
    if a_cubed > N:
        break
    
    current_power_val = a_cubed
    while current_power_val <= N:
        set_C2_candidates.add(current_power_val)
        
        # Prepare for next power: current_power_val * a
        # Check if current_power_val * a would exceed N.
        # current_power_val > N // a implies current_power_val * a > N.
        if current_power_val > N // a: 
            break # Exit inner loop (powers of a)
        current_power_val *= a
    
    a += 1

# C2: Count numbers in set_C2_candidates that are NOT perfect squares.
count_C2 = 0
for p_val in set_C2_candidates:
    # Check if p_val is a perfect square
    sqrt_pval = math.isqrt(p_val) # floor(sqrt(p_val))
    if sqrt_pval * sqrt_pval != p_val: # if p_val is not a perfect square
        count_C2 += 1
        
total_ans = count_C1 + count_C2
print(total_ans)