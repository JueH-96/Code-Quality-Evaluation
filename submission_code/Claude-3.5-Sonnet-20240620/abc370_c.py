# YOUR CODE HERE
def find_min_operations(S, T):
    result = []
    while S != T:
        for i in range(len(S)):
            if S[i] != T[i]:
                S = S[:i] + T[i] + S[i+1:]
                result.append(S)
                break
    return result

S = input().strip()
T = input().strip()

operations = find_min_operations(S, T)
print(len(operations))
for op in operations:
    print(op)