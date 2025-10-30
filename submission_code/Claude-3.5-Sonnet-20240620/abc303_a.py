# YOUR CODE HERE
def are_similar(a, b):
    return (a == b) or (a in '1l' and b in '1l') or (a in '0o' and b in '0o')

N = int(input())
S = input().strip()
T = input().strip()

if all(are_similar(s, t) for s, t in zip(S, T)):
    print("Yes")
else:
    print("No")