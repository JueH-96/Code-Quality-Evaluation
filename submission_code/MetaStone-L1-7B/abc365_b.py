n = int(input())
a = list(map(int, input().split()))
sorted_a = sorted(a, reverse=True)
second_largest = sorted_a[1]
index = a.index(second_largest)
print(index + 1)