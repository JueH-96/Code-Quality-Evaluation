# YOUR CODE HERE
S = input().strip()
number = int(S[3:])

if 1 <= number <= 314 or 317 <= number <= 349:
    print("Yes")
else:
    print("No")