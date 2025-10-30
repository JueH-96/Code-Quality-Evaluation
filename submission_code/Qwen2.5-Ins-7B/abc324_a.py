# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
first = a[0]
for i in a:
    if i != first:
        print("No")
        exit()
print("Yes")