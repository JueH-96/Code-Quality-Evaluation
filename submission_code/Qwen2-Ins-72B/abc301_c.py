S = list(input())
T = list(input())

S_a = S.count('@')
T_a = T.count('@')
S_ = [s for s in S if s != '@']
T_ = [t for t in T if t != '@']

if S_a + T_a < 7:
    if S_ != T_:
        print('No')
        exit()
    else:
        print('Yes')
        exit()

S_.extend(['a', 't', 'c', 'o', 'd', 'e', 'r'][:S_a])
T_.extend(['a', 't', 'c', 'o', 'd', 'e', 'r'][:T_a])

S_.sort()
T_.sort()

if S_ == T_:
    print('Yes')
else:
    print('No')