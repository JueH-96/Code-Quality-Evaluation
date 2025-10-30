def compute_sum(arr):
    arr.sort()
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]
    total = 0
    for i in range(n):
        total += arr[i] * i - prefix[i]
    return total

def calculate_contribution(group):
    if len(group) < 2:
        return 0
    u = []
    v = []
    for x, y in group:
        u.append(x + y)
        v.append(x - y)
    sum_u = compute_sum(u)
    sum_v = compute_sum(v)
    return (sum_u + sum_v) // 2

n = int(input())
group0 = []
group1 = []
for _ in range(n):
    x, y = map(int, input().split())
    if (x + y) % 2 == 0:
        group0.append((x, y))
    else:
        group1.append((x, y))

total = calculate_contribution(group0) + calculate_contribution(group1)
print(total)