n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
current_sum = 0
total_sum = 0

for i in range(n):
    num = a[i]
    p = last_occurrence.get(num, -1)
    current_sum += i - p
    total_sum += current_sum
    last_occurrence[num] = i

print(total_sum)