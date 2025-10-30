# YOUR CODE HERE
n = int(input())
s = []
for _ in range(n):
    s.append(input())

for i in range(n - 1):
    if s[i] == 'sweet' and s[i+1] == 'sweet':
        print('No')
        exit()

print('Yes')