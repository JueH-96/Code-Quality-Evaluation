import sys

S, T = input().split()

found = False
for w in range(1, len(S)):
    for c in range(1, w + 1):
        result = []
        pos = 0
        while pos < len(S):
            chunk = S[pos:pos + w]
            if len(chunk) >= c:
                result.append(chunk[c-1])
            pos += w
        if ''.join(result) == T:
            print("Yes")
            sys.exit()

print("No")