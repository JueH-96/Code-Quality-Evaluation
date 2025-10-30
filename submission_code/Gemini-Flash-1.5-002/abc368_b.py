# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

count = 0
while len([x for x in a if x > 0]) > 1:
    a.sort(reverse=True)
    a[0] -= 1
    a[1] -= 1
    count += 1

print(count)