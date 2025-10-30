# YOUR CODE HERE

S = input()

if S[3:] == "316":
    print("No")
elif int(S[3:]) <= 349:
    print("Yes")
else:
    print("No")