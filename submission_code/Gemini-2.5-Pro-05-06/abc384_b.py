# Read the number of contests (N) and initial rating (R)
N, R = map(int, input().split())

current_rating = R

# Process each contest
for _ in range(N):
    # Read contest division (D) and performance (A)
    D, A = map(int, input().split())

    subject_to_update = False  # Flag to track if rating should be updated

    if D == 1:
        # ARC Div. 1: rating update if current_rating is between 1600 and 2799 inclusive
        if 1600 <= current_rating <= 2799:
            subject_to_update = True
    else:  # D == 2, as D is guaranteed to be 1 or 2
        # ARC Div. 2: rating update if current_rating is between 1200 and 2399 inclusive
        if 1200 <= current_rating <= 2399:
            subject_to_update = True
    
    # If subject to update, change rating by performance A
    if subject_to_update:
        current_rating += A
        
# Print the final rating
print(current_rating)