# YOUR CODE HERE
def is_subject_to_update(rating, division):
    if division == 1:
        return 1600 <= rating <= 2799
    else:  # division == 2
        return 1200 <= rating <= 2399

# Read input
N, R = map(int, input().split())
contests = [list(map(int, input().split())) for _ in range(N)]

# Process contests
for D, A in contests:
    if is_subject_to_update(R, D):
        R += A

# Print final rating
print(R)