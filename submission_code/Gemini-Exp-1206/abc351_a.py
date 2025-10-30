a = list(map(int, input().split()))
b = list(map(int, input().split()))

takahashi_score = sum(a)
aoki_score = sum(b)

needed_runs = takahashi_score - aoki_score + 1

print(needed_runs)