n = int(input())
a = list(map(int, input().split()))
sorted_desc = sorted(a, reverse=True)
second_largest = sorted_desc[1]
index = a.index(second_largest) + 1
print(index)