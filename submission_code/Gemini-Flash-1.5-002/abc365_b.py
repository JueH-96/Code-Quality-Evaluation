# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

max1 = -1
max2 = -1
max1_index = -1

for i in range(n):
    if a[i] > max1:
        max2 = max1
        max1 = a[i]
        max1_index = i + 1
    elif a[i] > max2 and a[i] != max1:
        max2 = a[i]
        
        
for i in range(n):
    if a[i] == max2:
        print(i+1)
        break