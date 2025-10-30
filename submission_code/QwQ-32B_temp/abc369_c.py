n = int(input())
if n == 1:
    print(1)
else:
    a = list(map(int, input().split()))
    diffs = []
    for i in range(n-1):
        diffs.append(a[i+1] - a[i])
    sum_runs = 0
    current_val = diffs[0]
    current_length = 1
    for d in diffs[1:]:
        if d == current_val:
            current_length += 1
        else:
            sum_runs += current_length * (current_length + 1) // 2
            current_val = d
            current_length = 1
    # Add the last run
    sum_runs += current_length * (current_length + 1) // 2
    total = n + sum_runs
    print(total)