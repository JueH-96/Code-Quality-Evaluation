n, x_needed = map(int, input().split())
current_scores = list(map(int, input().split()))
min_score = 101  # Initialize with a value higher than the maximum possible score

for trial in range(0, 101):
    all_scores = current_scores.copy()
    all_scores.append(trial)
    all_scores.sort()
    # Calculate the sum excluding the first and last elements
    total = sum(all_scores[1:-1])
    if total >= x_needed:
        if trial < min_score:
            min_score = trial

if min_score == 101:
    print(-1)
else:
    print(min_score)