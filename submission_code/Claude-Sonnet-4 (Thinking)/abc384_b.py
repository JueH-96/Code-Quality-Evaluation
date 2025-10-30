n, r = map(int, input().split())
rating = r

for _ in range(n):
    d, a = map(int, input().split())
    
    if d == 1:  # Div 1
        if 1600 <= rating <= 2799:
            rating += a
    else:  # Div 2
        if 1200 <= rating <= 2399:
            rating += a

print(rating)