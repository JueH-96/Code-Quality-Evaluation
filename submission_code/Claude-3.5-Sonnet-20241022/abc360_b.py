S = input()
T = input()

n = len(S)

# Try all possible values of w and c
for w in range(1, n):
    for c in range(1, w+1):
        # Split S into substrings of length w
        substrings = []
        for i in range(0, n, w):
            substrings.append(S[i:min(i+w, n)])
        
        # Get c-th character from each substring of length >= c
        result = ''
        for s in substrings:
            if len(s) >= c:
                result += s[c-1]
        
        # Check if result equals T
        if result == T:
            print("Yes")
            exit()

print("No")