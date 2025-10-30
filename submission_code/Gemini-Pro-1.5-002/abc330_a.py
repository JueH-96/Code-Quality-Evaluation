# YOUR CODE HERE
n, l = map(int, input().split())
a = list(map(int, input().split()))

count = 0
for score in a:
    if score >= l:
        count += 1

print(count)