# YOUR CODE HERE
def find_correctly_typed_characters(S, T):
    s_index = 0
    t_index = 0
    result = []
    
    while s_index < len(S) and t_index < len(T):
        if S[s_index] == T[t_index]:
            result.append(t_index + 1)
            s_index += 1
        t_index += 1
    
    return result

import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

result = find_correctly_typed_characters(S, T)
print(" ".join(map(str, result)))