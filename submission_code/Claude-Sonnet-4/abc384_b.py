# YOUR CODE HERE
n, r = map(int, input().split())

current_rating = r

for _ in range(n):
    d, a = map(int, input().split())
    
    # Check if rating update applies
    if d == 1:  # Div 1
        if 1600 <= current_rating <= 2799:
            current_rating += a
    else:  # Div 2
        if 1200 <= current_rating <= 2399:
            current_rating += a

print(current_rating)