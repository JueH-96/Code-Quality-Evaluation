n = int(input())
a = list(map(int, input().split()))

max1 = max(a)
without_max = [x for x in a if x != max1]
max2 = max(without_max)

index = a.index(max2) + 1

print(index)