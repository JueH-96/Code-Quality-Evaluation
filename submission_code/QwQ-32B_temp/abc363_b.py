N, T, P = map(int, input().split())
L = list(map(int, input().split()))

days_needed = [max(0, T - x) for x in L]
sorted_days = sorted(days_needed)
current_count = sum(1 for d in days_needed if d == 0)

if current_count >= P:
    print(0)
else:
    print(sorted_days[P - 1])