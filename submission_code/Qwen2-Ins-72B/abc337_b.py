S = input()

A = S.find('B')
B = S.rfind('B')
C = S.rfind('C')

if A == -1 or B == -1 or C == -1:
    if S == 'A' * len(S) or S == 'B' * len(S) or S == 'C' * len(S):
        print('Yes')
    else:
        print('No')
else:
    if A == B and B == C:
        print('No')
    elif A < B and B < C:
        print('Yes')
    else:
        print('No')