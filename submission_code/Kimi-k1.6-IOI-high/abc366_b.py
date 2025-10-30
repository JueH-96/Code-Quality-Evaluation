n = int(input())
strings = [input().strip() for _ in range(n)]
if n == 0:
    print()
else:
    M = max(len(s) for s in strings)
    result = []
    for j in range(1, M + 1):
        current = []
        for i in range(n-1, -1, -1):
            s = strings[i]
            if j <= len(s):
                current.append(s[j-1])
            else:
                current.append('*')
        t = ''.join(current).rstrip('*')
        result.append(t)
    for t in result:
        print(t)