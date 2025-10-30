N, X = map(int, input().split())
A = list(map(int, input().split()))

min_A = min(A)
max_A = max(A)
sum_A = sum(A)

best = None

# Case 1: S becomes the new minimum (S < min_A)
if min_A > 0:
    final_grade = sum_A - max_A
    if final_grade >= X:
        best = 0

# Case 2: S becomes the new maximum (S > max_A)
if max_A < 100:
    final_grade = sum_A - min_A
    if final_grade >= X:
        candidate = max_A + 1
        if best is None or candidate < best:
            best = candidate

# Case 3: S is between min and max (min_A <= S <= max_A)
needed = X - sum_A + min_A + max_A
if 0 <= needed <= 100 and min_A <= needed <= max_A:
    if best is None or needed < best:
        best = needed

if best is not None:
    print(best)
else:
    print(-1)