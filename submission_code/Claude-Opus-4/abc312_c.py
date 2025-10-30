# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the arrays
A.sort()
B.sort()

# We need to check all critical points
# Critical points are: all A_i values and all B_i + 1 values
critical_points = set()
for a in A:
    critical_points.add(a)
for b in B:
    critical_points.add(b + 1)

# Also check the minimum possible answer
critical_points.add(min(A))

# Convert to sorted list
critical_points = sorted(list(critical_points))

# For each critical point, check if it satisfies the condition
for x in critical_points:
    # Count sellers who can sell at price x (A_i <= x)
    sellers = 0
    for a in A:
        if a <= x:
            sellers += 1
    
    # Count buyers who can buy at price x (B_i >= x)
    buyers = 0
    for b in B:
        if b >= x:
            buyers += 1
    
    # Check if condition is satisfied
    if sellers >= buyers:
        print(x)
        break