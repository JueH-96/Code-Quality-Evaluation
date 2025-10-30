# YOUR CODE HERE

n = int(input())
a = list(map(int, input().split()))

a.sort()

i = 0
while i < n and a[i] <= i+1:
    i += 1

print(i)