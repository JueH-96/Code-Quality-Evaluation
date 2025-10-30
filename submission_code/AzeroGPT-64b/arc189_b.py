N = int(input())
X = list(map(int,input().split()))

Y = [0]*N
for i in range(N):
    Y[i] = X[i] - i//2 * (X[1]-X[0])
    
Y_1 = Y[0]
Y_N = Y[N-1]

D = Y_N - Y_1

Z = [0]*N
for i in range(N):
    Z[i] = D * i//2 + Y_1 + i

S = sum(Z)
print(S)