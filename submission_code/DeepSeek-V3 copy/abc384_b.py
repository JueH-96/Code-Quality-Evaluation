# Read input values
N, R = map(int, input().split())
contests = []
for _ in range(N):
    D, A = map(int, input().split())
    contests.append((D, A))

# Initialize current rating
current_rating = R

# Process each contest
for D, A in contests:
    if D == 1:
        # Div. 1: 1600 <= rating <= 2799
        if 1600 <= current_rating <= 2799:
            current_rating += A
    else:
        # Div. 2: 1200 <= rating <= 2399
        if 1200 <= current_rating <= 2399:
            current_rating += A

# Output the final rating
print(current_rating)