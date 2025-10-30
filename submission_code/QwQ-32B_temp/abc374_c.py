n = int(input())
k_list = list(map(int, input().split()))
sum_S = sum(k_list)

possible = {0}
for num in k_list:
    temp = set()
    for s in possible:
        temp.add(s + num)
    possible.update(temp)

min_max = float('inf')
for a in possible:
    current_max = max(a, sum_S - a)
    if current_max < min_max:
        min_max = current_max

print(min_max)