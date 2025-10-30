import sys
data = sys.stdin.read().strip()
parts = data.split()
S = parts[0]
T = parts[1]
len_S = len(S)
found = False
for w in range(1, len_S):
    for c in range(1, w + 1):
        result = ""
        for i in range(0, len_S, w):
            substr = S[i : i + w]
            if len(substr) >= c:
                result += substr[c - 1]
        if result == T:
            found = True
            break
    if found:
        break
if found:
    print("Yes")
else:
    print("No")