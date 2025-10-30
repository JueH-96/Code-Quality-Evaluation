t = int(input().strip())
results = []
for _ in range(t):
    s = input().strip()
    if s == "abc":
        results.append("YES")
    else:
        s1 = s[1] + s[0] + s[2]  # Swap first and second characters
        s2 = s[2] + s[1] + s[0]  # Swap first and third characters
        s3 = s[0] + s[2] + s[1]  # Swap second and third characters
        if "abc" in [s1, s2, s3]:
            results.append("YES")
        else:
            results.append("NO")

for res in results:
    print(res)