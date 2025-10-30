# YOUR CODE HERE

N = int(input())

takahashi_total = 0
aoki_total = 0

for i in range(N):
    X, Y = map(int, input().split())
    takahashi_total += X
    aoki_total += Y

if takahashi_total > aoki_total:
    print("Takahashi")
elif takahashi_total < aoki_total:
    print("Aoki")
else:
    print("Draw")