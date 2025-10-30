n = int(input())
a = list(map(int, input().split()))
s = input().strip()

# Precompute mex for all possible (a, e, x) combinations
mex_dict = {}
for a_val in [0, 1, 2]:
    for e_val in [0, 1, 2]:
        for x_val in [0, 1, 2]:
            seen = {a_val, e_val, x_val}
            mex = 0
            while mex in seen:
                mex += 1
            mex_dict[(a_val, e_val, x_val)] = mex

# Precompute prefix sums for M's
m0 = [0] * (n + 1)
m1 = [0] * (n + 1)
m2 = [0] * (n + 1)

for j in range(n):
    current_char = s[j]
    current_a = a[j]
    m0[j+1] = m0[j]
    m1[j+1] = m1[j]
    m2[j+1] = m2[j]
    if current_char == 'M':
        if current_a == 0:
            m0[j+1] += 1
        elif current_a == 1:
            m1[j+1] += 1
        else:
            m2[j+1] += 1

# Precompute suffix sums for X's
x0 = [0] * (n + 1)
x1 = [0] * (n + 1)
x2 = [0] * (n + 1)

for j in range(n-1, -1, -1):
    current_char = s[j]
    current_a = a[j]
    x0[j] = x0[j+1]
    x1[j] = x1[j+1]
    x2[j] = x2[j+1]
    if current_char == 'X':
        if current_a == 0:
            x0[j] += 1
        elif current_a == 1:
            x1[j] += 1
        else:
            x2[j] += 1

# Calculate the total sum
total = 0
for j in range(n):
    if s[j] == 'E':
        e_val = a[j]
        m_counts = [m0[j], m1[j], m2[j]]
        x_counts = [x0[j+1], x1[j+1], x2[j+1]]
        for a_val in [0, 1, 2]:
            for x_val in [0, 1, 2]:
                count = m_counts[a_val] * x_counts[x_val]
                if count == 0:
                    continue
                total += count * mex_dict[(a_val, e_val, x_val)]

print(total)