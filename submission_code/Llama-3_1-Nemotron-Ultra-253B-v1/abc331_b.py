n, s, m, l = map(int, input().split())
min_cost = float('inf')
max_l = (n + 11) // 12
for l_count in range(max_l + 1):
    rem = max(0, n - 12 * l_count)
    if rem == 0:
        cost_86 = 0
    else:
        m_max = (rem + 7) // 8
        min_sub = float('inf')
        for m_count in range(m_max + 1):
            required = rem - 8 * m_count
            s_count = 0 if required <= 0 else (required + 5) // 6
            current_cost = m_count * m + s_count * s
            if current_cost < min_sub:
                min_sub = current_cost
        cost_86 = min_sub
    total = l_count * l + cost_86
    if total < min_cost:
        min_cost = total
print(min_cost)