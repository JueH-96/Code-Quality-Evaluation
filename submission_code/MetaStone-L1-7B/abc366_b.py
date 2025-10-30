n = int(input())
strings = [input().strip() for _ in range(n)]
M = max(len(s) for s in strings)

# Initialize the result list
result = []

for j in range(1, M + 1):
    t = []
    for i in range(n-1, -1, -1):  # i from N-1 downto 0 (0-based)
        s = strings[i]
        if j <= len(s):
            t.append(s[j-1])
        else:
            t.append('*')
    result.append(''.join(t))

for line in result:
    print(line)