N = int(input())
A = list(map(int, input().split()))

max_exp = 0
curr_exp = 0
for i in range(N):
    if i % 2 == 0:
        curr_exp += 2 * A[i]
    else:
        curr_exp += A[i]
    max_exp = max(max_exp, curr_exp)

print(max_exp)