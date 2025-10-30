# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))
A.sort()

non_A_pairs = N - K
total_pairs = (2 * N - K) // 2
remaining_pairs = total_pairs - non_A_pairs

if remaining_pairs <= 0:
    print(0)
else:
    P = remaining_pairs
    selected = A[:2 * P]
    total = 0
    for i in range(0, 2 * P, 2):
        total += selected[i+1] - selected[i]
    print(total)