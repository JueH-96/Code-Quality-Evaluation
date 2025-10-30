# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))
s = set(a)
total_sum = k * (k + 1) // 2
sum_in_a = 0
for x in s:
    if 1 <= x <= k:
        sum_in_a += x
print(total_sum - sum_in_a)