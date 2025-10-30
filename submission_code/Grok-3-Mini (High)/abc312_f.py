import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
pull = []
regular = []
opener = []
for _ in range(N):
    T = int(data[index])
    index += 1
    X = int(data[index])
    index += 1
    if T == 0:
        pull.append(X)
    elif T == 1:
        regular.append(X)
    elif T == 2:
        opener.append(X)
# Sort all lists in descending order
pull.sort(reverse=True)
regular.sort(reverse=True)
opener.sort(reverse=True)
# Compute prefix sums
num_P = len(pull)
P_sum = [0]
cum_p = 0
for val in pull:
    cum_p += val
    P_sum.append(cum_p)

num_R = len(regular)
R_sum = [0]
cum_r = 0
for val in regular:
    cum_r += val
    R_sum.append(cum_r)

num_O = len(opener)
O_cap_sum = [0]
cum_o = 0
for val in opener:
    cum_o += val
    O_cap_sum.append(cum_o)

# Now, iterate over possible U
max_hap = 0
max_U_val = min(num_R, M)
for U in range(0, max_U_val + 1):
    if O_cap_sum[num_O] < U:
        continue  # Cannot achieve capacity U
    # Binary search for smallest K such that O_cap_sum[K] >= U
    left = 0
    right = num_O
    while left < right:
        mid = (left + right) // 2
        if O_cap_sum[mid] >= U:
            right = mid
        else:
            left = mid + 1
    K_min = left  # Should satisfy O_cap_sum[K_min] >= U
    R_rem = M - U - K_min
    if R_rem >= 0:
        A_pull = min(R_rem, num_P)
        hap = R_sum[U] + P_sum[A_pull]
        if hap > max_hap:
            max_hap = hap

# After checking all U, output the max happiness
print(max_hap)