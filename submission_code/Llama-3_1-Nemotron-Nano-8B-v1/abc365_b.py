n = int(input())
a = list(map(int, input().split()))
max1 = max(a)
filtered = [x for x in a if x != max1]
max2 = max(filtered)
print(a.index(max2) + 1)