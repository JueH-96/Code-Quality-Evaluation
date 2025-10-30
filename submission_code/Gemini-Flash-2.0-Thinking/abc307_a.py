n = int(input())
a = list(map(int, input().split()))

weekly_sums = []
for i in range(n):
    start_index = i * 7
    end_index = (i + 1) * 7
    weekly_sum = sum(a[start_index:end_index])
    weekly_sums.append(weekly_sum)

print(*weekly_sums)