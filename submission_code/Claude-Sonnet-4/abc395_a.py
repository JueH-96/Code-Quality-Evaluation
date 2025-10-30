# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

is_strictly_increasing = True
for i in range(n - 1):
    if a[i] >= a[i + 1]:
        is_strictly_increasing = False
        break

if is_strictly_increasing:
    print("Yes")
else:
    print("No")