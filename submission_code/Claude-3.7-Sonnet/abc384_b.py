# YOUR CODE HERE
N, R = map(int, input().split())
rating = R

for _ in range(N):
    D, A = map(int, input().split())
    
    # Check if Takahashi is subject to rating update
    if D == 1 and 1600 <= rating <= 2799:
        rating += A
    elif D == 2 and 1200 <= rating <= 2399:
        rating += A
        
print(rating)