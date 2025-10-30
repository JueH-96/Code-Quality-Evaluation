n = int(input())
A = list(map(int, input().split()))
current_sum = 0
max_neg_prefix = 0
sum_N = 0

for a in A:
    current_sum += a
    sum_N += a
    if -current_sum > max_neg_prefix:
        max_neg_prefix = -current_sum

S = max(0, max_neg_prefix)
current_passengers = S + sum_N
print(current_passengers)