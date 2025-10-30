# YOUR CODE HERE
S = input().strip()
if S[:3] != 'ABC':
    print("No")
else:
    num = int(S[3:])
    if 1 <= num <= 349 and num != 316:
        print("Yes")
    else:
        print("No")