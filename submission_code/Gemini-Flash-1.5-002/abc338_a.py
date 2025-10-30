# YOUR CODE HERE
s = input()
if len(s) == 0:
    print("No")
else:
    if 'A' <= s[0] <= 'Z' and all('a' <= c <= 'z' for c in s[1:]):
        print("Yes")
    else:
        print("No")