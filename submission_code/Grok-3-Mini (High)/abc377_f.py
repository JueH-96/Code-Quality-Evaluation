import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

row_att = set()
col_att = set()
sum_att = set()
diff_att = set()

for _ in range(M):
    a = int(data[index])
    index += 1
    b = int(data[index])
    index += 1
    row_att.add(a)
    col_att.add(b)
    sum_att.add(a + b)
    diff_att.add(a - b)

# Compute sum of singles
sum_A = len(row_att) * N
sum_B = len(col_att) * N
sum_C = 0
for s in sum_att:
    size_s = min(s - 1, 2 * N + 1 - s)
    sum_C += size_s
sum_D = 0
for d in diff_att:
    size_d = N - abs(d)
    sum_D += size_d

# Compute pair intersections
pair_AB = len(row_att) * len(col_att)
pair_AC = 0
for x in row_att:
    for s in sum_att:
        y_val = s - x
        if 1 <= y_val <= N:
            pair_AC += 1
pair_AD = 0
for x in row_att:
    for d in diff_att:
        y_val = x - d
        if 1 <= y_val <= N:
            pair_AD += 1
pair_BC = 0
for y in col_att:
    for s in sum_att:
        x_val = s - y
        if 1 <= x_val <= N:
            pair_BC += 1
pair_BD = 0
for y in col_att:
    for d in diff_att:
        x_val = y + d
        if 1 <= x_val <= N:
            pair_BD += 1
pair_CD = 0
for s in sum_att:
    for d in diff_att:
        if (s + d) % 2 == 0:
            x_val = (s + d) // 2
            y_val = (s - d) // 2
            if 1 <= x_val <= N and 1 <= y_val <= N:
                pair_CD += 1
sum_pair_inter = pair_AB + pair_AC + pair_AD + pair_BC + pair_BD + pair_CD

# Compute triple intersections
triple_ABC = 0
for x in row_att:
    for y in col_att:
        sum_xy = x + y
        if sum_xy in sum_att:
            triple_ABC += 1
triple_ABD = 0
for x in row_att:
    for y in col_att:
        diff_xy = x - y
        if diff_xy in diff_att:
            triple_ABD += 1
triple_ACD = 0
for s in sum_att:
    for d in diff_att:
        if (s + d) % 2 == 0:
            x_val = (s + d) // 2
            y_val = (s - d) // 2
            if 1 <= x_val <= N and 1 <= y_val <= N and x_val in row_att:
                triple_ACD += 1
triple_BCD = 0
for s in sum_att:
    for d in diff_att:
        if (s + d) % 2 == 0:
            x_val = (s + d) // 2
            y_val = (s - d) // 2
            if 1 <= x_val <= N and 1 <= y_val <= N and y_val in col_att:
                triple_BCD += 1
sum_triple = triple_ABC + triple_ABD + triple_ACD + triple_BCD

# Compute quadruple intersection
quad_ABCD = 0
for x in row_att:
    for y in col_att:
        if (x + y) in sum_att and (x - y) in diff_att:
            quad_ABCD += 1

# Compute union size using inclusion-exclusion
union_size = (sum_A + sum_B + sum_C + sum_D) - sum_pair_inter + sum_triple - quad_ABCD

# Total squares and safe squares
total_squares = N * N
safe_squares = total_squares - union_size

# Output the result
print(safe_squares)