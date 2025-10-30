N, T, P = map(int, input().split())
L = list(map(int, input().split()))

days_needed = [max(0, T - l) for l in L]
days_needed_sorted = sorted(days_needed)

print(days_needed_sorted[P-1])