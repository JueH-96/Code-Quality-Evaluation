# YOUR CODE HERE
S, T = input().split()

n = len(S)
found = False

# Try all possible values of w and c
for w in range(1, n):  # w must be < |S|
    for c in range(1, w + 1):  # c must be <= w
        # Split S into chunks of length w
        chunks = []
        for i in range(0, n, w):
            chunk = S[i:i+w]
            chunks.append(chunk)
        
        # Get c-th character from chunks with length >= c
        result = ""
        for chunk in chunks:
            if len(chunk) >= c:
                result += chunk[c-1]  # c-th character (1-indexed, so c-1 for 0-indexed)
        
        if result == T:
            found = True
            break
    
    if found:
        break

if found:
    print("Yes")
else:
    print("No")