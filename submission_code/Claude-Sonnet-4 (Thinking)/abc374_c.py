n = int(input())
k = list(map(int, input().split()))

total_sum = sum(k)
min_max = float('inf')

for mask in range(1 << n):
    group_a_sum = 0
    for i in range(n):
        if mask & (1 << i):
            group_a_sum += k[i]
    
    group_b_sum = total_sum - group_a_sum
    max_people = max(group_a_sum, group_b_sum)
    min_max = min(min_max, max_people)

print(min_max)