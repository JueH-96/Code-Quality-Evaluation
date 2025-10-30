n = int(input())
a = list(map(int, input().split()))

if n == 1:
    print(1)
else:
    diffs = []
    for i in range(n-1):
        diffs.append(a[i+1] - a[i])
    
    total = 0
    current_val = diffs[0]
    run_length = 1
    
    for i in range(1, len(diffs)):
        if diffs[i] == current_val:
            run_length += 1
        else:
            total += run_length * (run_length + 1) // 2
            current_val = diffs[i]
            run_length = 1
    # Add the last run
    total += run_length * (run_length + 1) // 2
    
    print(total + n)