n = int(input())
s = [input() for _ in range(n)]
max_len = max(len(x) for x in s)
for i in range(max_len):
    for j in range(n-1, -1, -1):
        if i < len(s[j]):
            print(s[j][i], end="")
        else:
            print("*", end="")
    print()