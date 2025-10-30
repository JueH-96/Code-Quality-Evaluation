t = int(input().strip())
results = []
for _ in range(t):
    s = input().strip()
    if s == "abc":
        results.append("YES")
    else:
        swap01 = s[1] + s[0] + s[2]
        swap02 = s[2] + s[1] + s[0]
        swap12 = s[0] + s[2] + s[1]
        if swap01 == "abc" or swap02 == "abc" or swap12 == "abc":
            results.append("YES")
        else:
            results.append("NO")

for res in results:
    print(res)