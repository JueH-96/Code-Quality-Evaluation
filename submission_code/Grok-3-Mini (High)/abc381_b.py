# YOUR CODE HERE
S = input().strip()
if len(S) % 2 != 0:
    print("No")
elif any(S[i] != S[i+1] for i in range(0, len(S), 2)):
    print("No")
else:
    if len(set(S)) == len(S) // 2:
        print("Yes")
    else:
        print("No")