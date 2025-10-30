# YOUR CODE HERE
n, l = map(int, input().split())
a = list(map(int, input().split()))
count = 0
for num in a:
    if num >= l:
        count += 1
print(count)