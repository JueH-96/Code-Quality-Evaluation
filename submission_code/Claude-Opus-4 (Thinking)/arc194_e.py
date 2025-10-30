from math import gcd

N, X, Y = map(int, input().split())
S = input().strip()
T = input().strip()

g = gcd(X, Y)

# Check for each residue class modulo g
for i in range(g):
    # Count zeros in positions congruent to i mod g
    s_zeros = sum(1 for j in range(i, N, g) if S[j] == '0')
    t_zeros = sum(1 for j in range(i, N, g) if T[j] == '0')
    
    if s_zeros != t_zeros:
        print("No")
        exit()

print("Yes")