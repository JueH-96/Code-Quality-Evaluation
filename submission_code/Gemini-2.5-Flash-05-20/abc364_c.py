# YOUR CODE HERE
import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    dishes = []
    for i in range(N):
        dishes.append((A[i], B[i]))

    # Initialize min_dishes. In the worst case, he eats all N dishes.
    # This happens if neither X nor Y thresholds are crossed even after eating all N dishes.
    min_dishes = N 

    # Strategy 1: Sort by sweetness (A_i) in descending order
    # This aims to exceed X as quickly as possible.
    dishes_sorted_by_A = sorted(dishes, key=lambda d: d[0], reverse=True)
    
    current_sweetness_sum = 0
    current_saltiness_sum = 0
    
    for i in range(N):
        current_sweetness_sum += dishes_sorted_by_A[i][0]
        current_saltiness_sum += dishes_sorted_by_A[i][1]
        
        # Check if the stopping condition is met for this prefix of dishes
        if current_sweetness_sum > X or current_saltiness_sum > Y:
            min_dishes = min(min_dishes, i + 1)
            break # Found the minimum for this particular sorting strategy, no need to check more dishes
    
    # Strategy 2: Sort by saltiness (B_i) in descending order
    # This aims to exceed Y as quickly as possible.
    dishes_sorted_by_B = sorted(dishes, key=lambda d: d[1], reverse=True)
    
    current_sweetness_sum = 0
    current_saltiness_sum = 0
    
    for i in range(N):
        current_sweetness_sum += dishes_sorted_by_B[i][0]
        current_saltiness_sum += dishes_sorted_by_B[i][1]
        
        # Check if the stopping condition is met for this prefix of dishes
        if current_sweetness_sum > X or current_saltiness_sum > Y:
            min_dishes = min(min_dishes, i + 1)
            break # Found the minimum for this particular sorting strategy, no need to check more dishes
            
    print(min_dishes)

solve()