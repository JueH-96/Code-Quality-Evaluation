n = int(input())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))

current_element = 1
for i in range(n):
    if current_element >= i + 1:
        current_element = a[current_element - 1][i]
    else:
        current_element = a[i][current_element - 1]

print(current_element)