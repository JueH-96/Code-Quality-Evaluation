# YOUR CODE HERE
S = input().strip()
if S[:3] != 'ABC':
    print("No")
else:
    num = int(S[3:])
    if num < 1 or num > 349:
        print("No")
    elif num == 316:
        print("No")
    else:
        print("Yes")