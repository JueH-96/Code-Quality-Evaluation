# Read the input
N, R = map(int, input().split())
ratings = []
for _ in range(N):
    D, A = map(int, input().split())
    ratings.append((D, A))

# Simulate the rating changes
for div, score in ratings:
    if 1600 <= R <= 2799 and div == 1:
        R += score
    elif 1200 <= R <= 2399 and div == 2:
        R += score

# Print the final rating
print(R)