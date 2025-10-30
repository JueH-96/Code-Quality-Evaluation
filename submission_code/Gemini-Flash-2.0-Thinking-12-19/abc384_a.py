N, c1, c2 = input().split()
N = int(N)
S = input()

result = []
for char in S:
    if char == c1:
        result.append(char)
    else:
        result.append(c2)

print("".join(result))