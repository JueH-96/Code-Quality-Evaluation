A = list(map(int, input().split()))
B = list(map(int, input().split()))

takahashi_score = sum(A)
aoki_score = sum(B)

needed_runs = takahashi_score - aoki_score + 1

print(needed_runs)