# YOUR CODE HERE
def is_airport_code(S, T):
    if T[-1] == 'X':
        T = T[:-1]
        S = S[:len(T)]
    for i in range(len(T)):
        if T[i] not in S[i:]:
            return 'No'
        S = S[S.index(T[i])+1:]
    return 'Yes'

S = input().strip()
T = input().strip()
print(is_airport_code(S, T))