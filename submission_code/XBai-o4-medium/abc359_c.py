sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

def compute_u_v(i, j):
    sum_ij = i + j
    u = sum_ij // 2
    temp_v = (i - j) - (sum_ij % 2)
    v = temp_v // 2
    return u, v

u1, v1 = compute_u_v(sx, sy)
u2, v2 = compute_u_v(tx, ty)

print(abs(u1 - u2) + abs(v1 - v2))