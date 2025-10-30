# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Sort in descending order to get strongest monsters first
a.sort(reverse=True)

max_points = 0

# Try defeating k monsters (k from 0 to n)
for k in range(n + 1):
    if k == 0:
        points = 0
    else:
        # Take the k strongest monsters
        selected = a[:k]
        points = 0
        
        # Assign monsters to positions optimally
        # Even positions (2nd, 4th, 6th defeated) get double points
        # Odd positions (1st, 3rd, 5th defeated) get single points
        
        # Number of even positions among k positions
        even_positions = k // 2
        # Number of odd positions among k positions  
        odd_positions = k - even_positions
        
        # Assign strongest monsters to even positions first (they get double points)
        for i in range(even_positions):
            points += 2 * selected[i]
        
        # Assign remaining monsters to odd positions
        for i in range(even_positions, k):
            points += selected[i]
    
    max_points = max(max_points, points)

print(max_points)