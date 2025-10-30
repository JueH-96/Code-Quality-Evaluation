S, T = input().split()
n = len(S)

found = False

# Try all possible values of w
for w in range(1, n):
    # Try all possible values of c
    for c in range(1, w + 1):
        # Split S into chunks of size w
        chunks = []
        for i in range(0, n, w):
            chunks.append(S[i:i+w])
        
        # Extract c-th character from chunks with length >= c
        result = ""
        for chunk in chunks:
            if len(chunk) >= c:
                result += chunk[c-1]  # c-1 because of 0-indexing
        
        # Check if result equals T
        if result == T:
            found = True
            break
    
    if found:
        break

if found:
    print("Yes")
else:
    print("No")