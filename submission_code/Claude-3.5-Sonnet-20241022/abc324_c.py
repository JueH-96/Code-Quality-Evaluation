def check_possible(T, S):
    # Case 1: Equal
    if T == S:
        return True
        
    # Case 2: Insertion
    if len(T) == len(S) + 1:
        for i in range(len(T)):
            if T[:i] + T[i+1:] == S:
                return True
                
    # Case 3: Deletion
    if len(S) == len(T) + 1:
        for i in range(len(S)):
            if S[:i] + S[i+1:] == T:
                return True
                
    # Case 4: Change one character
    if len(T) == len(S):
        diff = 0
        for i in range(len(T)):
            if T[i] != S[i]:
                diff += 1
        if diff == 1:
            return True
            
    return False

N, T = input().split()
N = int(N)
result = []

for i in range(N):
    S = input()
    if check_possible(T, S):
        result.append(i+1)

print(len(result))
if result:
    print(*result)