# YOUR CODE HERE
n = int(input())
takahashi = 0
aoki = 0
for _ in range(n):
    x, y = map(int, input().split())
    takahashi += x
    aoki += y

if takahashi > aoki:
    print("Takahashi")
elif aoki > takahashi:
    print("Aoki")
else:
    print("Draw")