# YOUR CODE HERE
N, c1, c2 = input().split()
S = input()
result = ''
for ch in S:
    if ch == c1:
        result += ch
    else:
        result += c2
print(result)