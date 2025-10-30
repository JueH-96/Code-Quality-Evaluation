from itertools import product

def matches(S, T):
    N, M = len(S), len(T)
    T_occurrences = []
    for i in range(N - M + 1):
        if S[i:i + M] == "##" * M:
            T_occurrences.append(i)
    T_patterns = [''.join(p) for p in product('ABCD', repeat=M)]

    for patt in T_patterns:
        for t_start in T_occurrences:
            X = list(S)
            X[t_start:t_start + M] = list(patt)
            if "".join(X) == T * (N // M) + T[:N % M]:
                return "Yes"
    return "No"

# Reading input
N, M = map(int, input().split())
S = input()
T = input()

# Converting S to a format where all characters are replaced by '#'
S = "#" * N
# Calling the function and printing the output
print(matches(S, T))