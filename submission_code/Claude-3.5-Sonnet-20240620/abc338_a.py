# YOUR CODE HERE
def is_capitalized(s):
    return s[0].isupper() and s[1:].islower()

S = input().strip()
print("Yes" if is_capitalized(S) else "No")