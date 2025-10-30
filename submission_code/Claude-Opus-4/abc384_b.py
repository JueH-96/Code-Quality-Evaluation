# YOUR CODE HERE
N, R = map(int, input().split())
rating = R

for _ in range(N):
    D, A = map(int, input().split())
    
    if D == 1:  # Div. 1
        if 1600 <= rating <= 2799:
            rating += A
    else:  # D == 2, Div. 2
        if 1200 <= rating <= 2399:
            rating += A

print(rating)