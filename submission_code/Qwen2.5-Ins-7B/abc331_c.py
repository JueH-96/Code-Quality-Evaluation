# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
s = sum(a)
count = [0] * (max(a) + 1)
for i in a:
    s -= i
    count[i] += 1
for i in a:
    s -= count[i] * i
    print(s, end=' ')
    s += i