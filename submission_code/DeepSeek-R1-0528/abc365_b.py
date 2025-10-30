n = int(input().strip())
arr = list(map(int, input().split()))

max_val = arr[0]
max_index = 1
second_max_val = 0
second_max_index = -1

for i in range(1, n):
    num = arr[i]
    if num > max_val:
        second_max_val = max_val
        second_max_index = max_index
        max_val = num
        max_index = i + 1
    elif num > second_max_val:
        second_max_val = num
        second_max_index = i + 1

print(second_max_index)