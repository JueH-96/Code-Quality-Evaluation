# YOUR CODE HERE
n, q = map(int, input().split())
s = input()
for _ in range(q):
    l, r = map(int, input().split())
    count = 0
    for i in range(l - 1, r - 1):
        if i + 1 < r and s[i] == s[i + 1]:
            count += 1
    print(count)