import sys
data = sys.stdin.read().split()
N = int(data[0])
takahashi_sum = sum(int(data[2 * i + 1]) for i in range(N))
aoki_sum = sum(int(data[2 * i + 2]) for i in range(N))
if takahashi_sum > aoki_sum:
    print("Takahashi")
elif aoki_sum > takahashi_sum:
    print("Aoki")
else:
    print("Draw")