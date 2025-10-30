N, T, P = map(int, input().split())
L = list(map(int, input().split()))

initial_count = sum(1 for x in L if x >= T)

if initial_count >= P:
    print(0)
else:
    required_days = [T - x for x in L if x < T]
    if not required_days:
        print(0)
    else:
        max_d = max(required_days)
        for d in range(1, max_d + 1):
            current_count = initial_count + sum(1 for x in L if x < T and (T - x) <= d)
            if current_count >= P:
                print(d)
                break