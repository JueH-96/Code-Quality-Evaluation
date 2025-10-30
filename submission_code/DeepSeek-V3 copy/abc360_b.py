# YOUR CODE HERE
S, T = input().split()
n = len(S)
m = len(T)
for w in range(1, n):
    substrings = [S[i:i+w] for i in range(0, n, w)]
    for c in range(1, w+1):
        extracted = []
        for s in substrings:
            if len(s) >= c:
                extracted.append(s[c-1])
        if ''.join(extracted) == T:
            print("Yes")
            exit()
print("No")