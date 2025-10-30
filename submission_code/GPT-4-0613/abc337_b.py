# YOUR CODE HERE
S = input().strip()
if S.count('A') + S.count('B') + S.count('C') != len(S):
    print("No")
elif S.find('A') > S.find('B') or S.find('B') > S.find('C'):
    print("No")
elif S.rfind('A') > S.rfind('B') or S.rfind('B') > S.rfind('C'):
    print("No")
else:
    print("Yes")