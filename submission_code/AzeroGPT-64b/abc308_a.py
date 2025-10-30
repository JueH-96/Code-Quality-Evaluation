# Inputs
S = list(map(int, input().split()))
# Conditions
if (all(S[i] <= S[i + 1] for i in range(7)) and
    all(100 <= S[i] <= 675 for i in range(8)) and
    all(S[i] % 25 == 0 for i in range(8))):
    print('Yes')
else:
    print('No')