# YOUR CODE HERE
N = int(input())
ranges = [list(map(int, input().split())) for _ in range(N)]
total_sum = sum(R for L, R in ranges)
if total_sum < 0:
    print("No")
    exit()
X = []
remaining_sum = 0
for L, R in ranges:
    if remaining_sum + L >= 0:
        X.append(-remaining_sum)
        remaining_sum = 0
    else:
        X.append(L)
        remaining_sum += L - L
if remaining_sum != 0:
    print("No")
else:
    print("Yes")
    print(*X)