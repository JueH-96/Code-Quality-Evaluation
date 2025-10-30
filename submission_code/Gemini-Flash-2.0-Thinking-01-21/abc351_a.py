a_runs_str = input().split()
a_runs = [int(x) for x in a_runs_str]
b_runs_str = input().split()
b_runs = [int(x) for x in b_runs_str]

takahashi_score = sum(a_runs)
aoki_score = sum(b_runs)

min_runs_to_win = takahashi_score - aoki_score + 1

print(min_runs_to_win)