n, m, l = map(int, input().split())
a = list(map(int, input().split()))  # a_1 to a_n, stored as a[0] to a[n-1]
b = list(map(int, input().split()))  # b_1 to b_m, stored as b[0] to b[m-1]

excluded = set()
for _ in range(l):
    c, d = map(int, input().split())
    excluded.add((c, d))

# Sort main dishes in descending order of their cost
main_dishes = sorted([(a[i], i + 1) for i in range(n)], key=lambda x: (-x[0], x[1]))
# Sort side dishes in descending order of their cost
side_dishes = sorted([(b[j], j + 1) for j in range(m)], key=lambda x: (-x[0], x[1]))

K = 200
top_main = main_dishes[:K]
top_side = side_dishes[:K]

max_sum = -float('inf')

# Check all combinations where main is in top_main and side is any
for m_entry in top_main:
    a_cost, a_dish = m_entry
    for j in range(m):
        b_dish = j + 1
        b_cost = b[j]
        if (a_dish, b_dish) not in excluded:
            current_sum = a_cost + b_cost
            if current_sum > max_sum:
                max_sum = current_sum

# Check all combinations where side is in top_side and main is any
for s_entry in top_side:
    b_cost, b_dish = s_entry
    for i in range(n):
        a_dish = i + 1
        a_cost = a[i]
        if (a_dish, b_dish) not in excluded:
            current_sum = a_cost + b_cost
            if current_sum > max_sum:
                max_sum = current_sum

print(max_sum)