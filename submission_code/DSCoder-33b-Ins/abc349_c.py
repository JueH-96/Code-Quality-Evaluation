# YOUR CODE HERE

S = input()
T = input()

def is_airport_code(S, T):
    if len(T) == 4 and T[-1] == 'X':
        T = T[:-1]
        if len(set(T) & set(S)) == len(T):
            return 'Yes'
    else:
        if len(set(T) & set(S)) == len(T):
            return 'Yes'
    return 'No'

print(is_airport_code(S, T))