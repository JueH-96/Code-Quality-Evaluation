n, k = map(int, input().split())
a = list(map(int, input().split()))
split_point = n - k
first_part = a[:split_point]
second_part = a[split_point:]
result = second_part + first_part
print(' '.join(map(str, result)))