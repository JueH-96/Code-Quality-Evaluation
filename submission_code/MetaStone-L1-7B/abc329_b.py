n = int(input())
a = list(map(int, input().split()))
max_val = max(a)
filtered = [x for x in a if x < max_val]
print(max(filtered))