n, x, y = map(int, input().split())
dishes = [tuple(map(int, input().split())) for _ in range(n)]

max_dishes = 0

for i in range(n):
    for j in range(n):
        if i == j:
            a_i, b_i = dishes[i]
            sum_a_initial = a_i
            sum_b_initial = b_i
            max_a = a_i
            max_b = b_i
            remaining_sum_a = x
            remaining_sum_b = y
            remaining = [dishes[k] for k in range(n) if k != i]
        else:
            a_i, b_i = dishes[i]
            a_j, b_j = dishes[j]
            sum_a_initial = a_i + a_j
            sum_b_initial = b_i + b_j
            max_a = max(a_i, a_j)
            max_b = max(b_i, b_j)
            remaining_sum_a = x + max_a - sum_a_initial
            remaining_sum_b = y + max_b - sum_b_initial
            if remaining_sum_a < 0 or remaining_sum_b < 0:
                continue
            remaining = [dishes[k] for k in range(n) if k != i and k != j]
        
        remaining.sort(key=lambda dish: dish[0] + dish[1])
        current_a = 0
        current_b = 0
        count = 0
        for a, b in remaining:
            if current_a + a <= remaining_sum_a and current_b + b <= remaining_sum_b:
                current_a += a
                current_b += b
                count += 1
        if i == j:
            total = 1 + count
        else:
            total = 2 + count
        if total > max_dishes:
            max_dishes = total

print(max_dishes)