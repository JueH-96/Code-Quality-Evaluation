S = input()
T = input()
N = min(len(S), len(T)) # Equal length portion
if S == T: # Already equal
    print(0)
else:
    for i in range(N):
        if S[i] != T[i]:
            print(i+1)
            break
    else: # equal up to N, but lengths not the same
        print(N+1)