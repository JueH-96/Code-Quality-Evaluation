N, c1, c2 = input().split()
N = int(N)
S = input()

result = ""
for char in S:
    if char != c1:
        result += c2
    else:
        result += char

print(result)