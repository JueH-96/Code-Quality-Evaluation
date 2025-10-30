# YOUR CODE HERE
n = int(input())
s = input()
count = 0
for i in range(n):
    length = 1
    while i + length <= n and s[i] == s[i + length]:
        length += 1
    count += length * (length - 1) // 2
print(count)