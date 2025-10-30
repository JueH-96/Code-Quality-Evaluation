# N P Q
N, P, Q = map(int, input().split())
# D_1 D_2 ... D_N
D = list(map(int, input().split()))

# Calculate the minimum total amount of money
if Q < P:
    print(min(P, Q + min(D)))
else:
    print(P)