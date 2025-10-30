arr = []
while True:
    num = int(input().strip())
    arr.append(num)
    if num == 0:
        break

for i in range(len(arr)-1, -1, -1):
    print(arr[i])