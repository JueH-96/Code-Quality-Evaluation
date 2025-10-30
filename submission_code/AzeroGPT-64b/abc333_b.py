S1S2, T1T2 = input(), input()

if S1S2 == T1T2 or S1S2[::-1] == T1T2 or set(S1S2) == set(T1T2) and S1S2 != T1T2 or {'B', 'D'} == set(S1S2) == set(T1T2):
    print('Yes')
else:
    print('No')