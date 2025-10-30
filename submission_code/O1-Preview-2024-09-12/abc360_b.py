S, T = input().split()
n = len(S)
found = False
for w in range(1, n):  # w from 1 to n-1
    for c in range(1, w+1):  # c from 1 to w
        substrings = [S[i:i+w] for i in range(0, n, w)]
        collected_chars = ''
        for substr in substrings:
            if len(substr) >= c:
                collected_chars += substr[c-1]
        if collected_chars == T:
            found = True
            break
    if found:
        break
if found:
    print("Yes")
else:
    print("No")