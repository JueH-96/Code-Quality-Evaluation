N = int(input().strip())
S = list(input().strip())
Q = int(input().strip())
for _ in range(Q):
    c, d = input().strip().split()
    S = [''.join([d if x == c else x for x in s]) for s in S]
print(''.join(S))