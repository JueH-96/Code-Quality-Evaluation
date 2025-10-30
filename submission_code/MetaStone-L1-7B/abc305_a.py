n = int(input())
lower = (n // 5) * 5
upper = lower + 5
if upper > 100:
    upper = 100

distance_lower = n - lower
distance_upper = upper - n

if distance_lower < distance_upper:
    print(lower)
else:
    print(upper)