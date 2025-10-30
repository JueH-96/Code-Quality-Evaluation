# YOUR CODE HERE
N = int(input())
takahashi_total = 0
aoki_total = 0

for _ in range(N):
    x, y = map(int, input().split())
    takahashi_total += x
    aoki_total += y

if takahashi_total > aoki_total:
    print("Takahashi")
elif aoki_total > takahashi_total:
    print("Aoki")
else:
    print("Draw")