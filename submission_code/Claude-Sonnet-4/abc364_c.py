import sys
from itertools import combinations

def solve():
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    dishes = [(A[i], B[i]) for i in range(N)]
    
    min_dishes = N + 1
    
    # Try different sorting strategies
    # Strategy 1: Sort by sweetness descending
    dishes_by_sweet = sorted(dishes, key=lambda x: x[0], reverse=True)
    total_sweet, total_salt = 0, 0
    for i, (sweet, salt) in enumerate(dishes_by_sweet):
        total_sweet += sweet
        total_salt += salt
        if total_sweet > X or total_salt > Y:
            min_dishes = min(min_dishes, i + 1)
            break
    
    # Strategy 2: Sort by saltiness descending
    dishes_by_salt = sorted(dishes, key=lambda x: x[1], reverse=True)
    total_sweet, total_salt = 0, 0
    for i, (sweet, salt) in enumerate(dishes_by_salt):
        total_sweet += sweet
        total_salt += salt
        if total_sweet > X or total_salt > Y:
            min_dishes = min(min_dishes, i + 1)
            break
    
    # Strategy 3: Sort by sum descending
    dishes_by_sum = sorted(dishes, key=lambda x: x[0] + x[1], reverse=True)
    total_sweet, total_salt = 0, 0
    for i, (sweet, salt) in enumerate(dishes_by_sum):
        total_sweet += sweet
        total_salt += salt
        if total_sweet > X or total_salt > Y:
            min_dishes = min(min_dishes, i + 1)
            break
    
    # For small numbers, try all combinations
    for k in range(1, min(min_dishes, min(N, 20) + 1)):
        found = False
        for combo in combinations(dishes, k):
            total_sweet = sum(dish[0] for dish in combo)
            total_salt = sum(dish[1] for dish in combo)
            if total_sweet > X or total_salt > Y:
                min_dishes = min(min_dishes, k)
                found = True
                break
        if found:
            break
    
    print(min_dishes)

solve()