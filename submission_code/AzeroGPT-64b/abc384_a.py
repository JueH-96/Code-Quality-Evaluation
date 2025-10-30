readline = (lambda: input())
N, c1, c2 = readline().split()
S = list(readline().strip())
print("".join(map(lambda c : c if c==c1 else c2,S)))