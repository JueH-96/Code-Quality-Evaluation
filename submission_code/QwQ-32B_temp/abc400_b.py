N, M = map(int, input().split())
sum_total = 0
current_term = 1
exceed = False

for i in range(M + 1):
    sum_total += current_term
    if sum_total > 10**9:
        exceed = True
        break
    if i < M:
        current_term *= N

print("inf" if exceed else sum_total)