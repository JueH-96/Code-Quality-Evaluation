# YOUR CODE HERE
N = int(input())

M = 1
K = [N]
A = [[i for i in range(1, N+1)]]

print(M)
for k, a in zip(K, A):
    print(k, *a)

S = input()

X = int(S.index('1')) + 1 if '1' in S else 1
print(X)