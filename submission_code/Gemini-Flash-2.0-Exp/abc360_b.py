S, T = input().split()
n = len(S)
t = len(T)

for w in range(1, n):
    for c in range(1, w + 1):
        temp = ""
        for i in range(0, n, w):
            if i + w <= n:
                sub = S[i:i + w]
            else:
                sub = S[i:]
            
            if len(sub) >= c:
                temp += sub[c-1]
        
        if temp == T:
            print("Yes")
            exit()

print("No")