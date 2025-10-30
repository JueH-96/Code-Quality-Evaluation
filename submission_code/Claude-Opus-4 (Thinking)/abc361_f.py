n = int(input())
result_set = set()

# b starts from 2
for b in range(2, 64):  # 2^63 > 10^18, so this is sufficient
    a = 1
    while a ** b <= n:
        result_set.add(a ** b)
        a += 1

print(len(result_set))