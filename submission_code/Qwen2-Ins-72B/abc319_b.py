n = int(input())
s = ["-"] * (n + 1)
s[0] = "1"
for i in range(2, 10):
    if n % i == 0:
        d = n // i
        for j in range(d, n + 1, d):
            if s[j] == "-":
                s[j] = str(i)
print("".join(s))