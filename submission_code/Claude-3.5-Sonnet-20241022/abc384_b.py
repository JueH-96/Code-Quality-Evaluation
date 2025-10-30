# Read input
N, R = map(int, input().split())
contests = []
for _ in range(N):
    d, a = map(int, input().split())
    contests.append((d, a))

# Current rating
rating = R

# Process each contest
for div, perf in contests:
    # Check if rating update applies based on division
    if div == 1:
        # Div 1: 1600-2799
        if 1600 <= rating <= 2799:
            rating += perf
    else:
        # Div 2: 1200-2399
        if 1200 <= rating <= 2399:
            rating += perf

# Print final rating
print(rating)