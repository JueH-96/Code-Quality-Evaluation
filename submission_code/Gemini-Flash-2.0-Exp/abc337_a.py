N = int(input())
takahashi = 0
aoki = 0
for i in range(N):
    X, Y = map(int, input().split())
    takahashi += X
    aoki += Y

if takahashi > aoki:
    print("Takahashi")
elif aoki > takahashi:
    print("Aoki")
else:
    print("Draw")