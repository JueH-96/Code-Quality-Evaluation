N = int(input())
S = input()
T = input()

def are_similar(c1, c2):
    if c1 == c2:
        return True
    if {c1, c2} == {'1', 'l'}:
        return True
    if {c1, c2} == {'0', 'o'}:
        return True
    return False

if all(are_similar(S[i], T[i]) for i in range(N)):
    print("Yes")
else:
    print("No")