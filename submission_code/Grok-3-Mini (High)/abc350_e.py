import sys

# Read input
data = sys.stdin.read().strip()
N, A, X, Y = map(int, data.split())

# Compute max exponents
def max_exp(base, N):
    exp = 0
    power = 1
    while power <= N:
        exp += 1
        power *= base
    return exp - 1

max_a_exp = max_exp(2, N)
max_b_exp = max_exp(3, N)
max_c_exp = max_exp(5, N)

# Generate m_set
m_set = set()
for a in range(0, max_a_exp + 1):
    for b in range(0, max_b_exp + 1):
        for c in range(0, max_c_exp + 1):
            k = (2 ** a) * (3 ** b) * (5 ** c)
            if k <= N:
                m = N // k
                m_set.add(m)

# Sort m_list
m_list = sorted(list(m_set))  # ascending order

# E_dict with E[0] = 0.0
E_dict = {0: 0.0}

# Compute E for each m in m_list
for m in m_list:
    # Compute val_fix
    floor_m_A = m // A
    val_fix = X + E_dict[floor_m_A]
    
    # Compute sum_b2to6
    sum_b2to6 = 0.0
    for b in range(2, 7):  # b=2,3,4,5,6
        floor_mb = m // b
        sum_b2to6 += E_dict[floor_mb]
    
    # Compute val_die
    val_die = (6.0 / 5.0) * Y + (1.0 / 5.0) * sum_b2to6
    
    # E_m is the minimum
    E_m = min(val_fix, val_die)
    
    # Store in E_dict
    E_dict[m] = E_m

# The answer is E_dict[N]
answer = E_dict[N]

# Print with high precision
print(f"{answer:.15f}")