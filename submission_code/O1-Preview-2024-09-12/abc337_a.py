# YOUR CODE HERE
N = int(input())
takahashi_total = 0
aoki_total = 0
for _ in range(N):
    X_i, Y_i = map(int, input().split())
    takahashi_total += X_i
    aoki_total += Y_i
if takahashi_total > aoki_total:
    print("Takahashi")
elif takahashi_total < aoki_total:
    print("Aoki")
else:
    print("Draw")