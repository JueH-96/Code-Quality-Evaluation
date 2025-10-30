# Read M and D
M_D = input().split()
M = int(M_D[0])
D = int(M_D[1])

# Read y, m, d
y_m_d = input().split()
y = int(y_m_d[0])
m = int(y_m_d[1])
d = int(y_m_d[2])

# Determine the next day
if d < D:
    d += 1
else:
    d = 1
    if m < M:
        m += 1
    else:
        m = 1
        y += 1

# Print the result
print(y, m, d)