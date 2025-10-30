import sys
data = sys.stdin.read().split()
N = int(data[0])
R = int(data[1])
C = int(data[2])
S_str = data[3]

# Compute cumulative sums
cum_r = [0] * (N + 1)
cum_c = [0] * (N + 1)
for t in range(1, N + 1):
    dir_char = S_str[t - 1]
    if dir_char == 'N':
        dr = -1
        dc = 0
    elif dir_char == 'W':
        dr = 0
        dc = -1
    elif dir_char == 'S':
        dr = 1
        dc = 0
    elif dir_char == 'E':
        dr = 0
        dc = 1
    cum_r[t] = cum_r[t - 1] + dr
    cum_c[t] = cum_c[t - 1] + dc

# Compute generation positions
gen_pos_to_time = {}  # dict with key (r,c), value gen time
gen_pos_to_time[(0, 0)] = 0  # initial generation at time 0
for g in range(1, N + 1):
    if (cum_r[g], cum_c[g]) not in gen_pos_to_time:
        gen_pos_to_time[(cum_r[g], cum_c[g])] = g

# Check for each time t from 1 to N
result = []
for t in range(1, N + 1):
    target_r = cum_r[t] - R
    target_c = cum_c[t] - C
    if (target_r, target_c) in gen_pos_to_time:
        g_gen = gen_pos_to_time[(target_r, target_c)]
        if g_gen <= t - 1:
            result.append('1')
        else:
            result.append('0')
    else:
        result.append('0')

# Output the result
print(''.join(result))