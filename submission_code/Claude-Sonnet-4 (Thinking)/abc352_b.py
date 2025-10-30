S = input().strip()
T = input().strip()

i = 0  # pointer for S
j = 0  # pointer for T
result = []

while i < len(S):
    # Find the next correct character
    while T[j] != S[i]:
        j += 1
    
    # Found the correct character
    result.append(j + 1)  # 1-indexed
    i += 1
    j += 1

print(' '.join(map(str, result)))