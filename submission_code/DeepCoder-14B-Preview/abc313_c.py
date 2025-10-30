n = int(input())
a = list(map(int, input().split()))
sum_a = sum(a)
m = sum_a // n
r = sum_a % n

initial_sum = sum(x - m for x in a if x > m)
cnt = sum(1 for x in a if x >= m + 1)
x = min(r, cnt)
print(initial_sum - x)