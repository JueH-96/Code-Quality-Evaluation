# Read the input strings S and T
S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    current = S
    X = []
    while current != T:
        candidates = []
        for i in range(len(current)):
            if current[i] != T[i]:
                # Create a new string by changing the i-th character to T[i]
                new_str = current[:i] + T[i] + current[i+1:]
                candidates.append(new_str)
        # If no candidates found (shouldn't happen as current != T)
        if not candidates:
            break
        # Sort the candidates lexicographically and choose the smallest
        candidates.sort()
        chosen = candidates[0]
        X.append(chosen)
        current = chosen
    # Output the result
    print(len(X))
    for s in X:
        print(s)