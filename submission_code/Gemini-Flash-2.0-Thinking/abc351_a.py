a_str = input()
b_str = input()

a = [int(x) for x in a_str.split()]
b = [int(x) for x in b_str.split()]

takahashi_score = sum(a)
aoki_score_8 = sum(b)

min_runs = takahashi_score - aoki_score_8 + 1

print(min_runs)