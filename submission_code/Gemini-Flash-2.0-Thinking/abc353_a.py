n = int(input())
h = list(map(int, input().split()))

first_height = h[0]
found_index = -1

for i in range(1, n):
    if h[i] > first_height:
        found_index = i + 1
        break

print(found_index)