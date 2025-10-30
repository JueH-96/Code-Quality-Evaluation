def min_distance(a, b, n):
    """Calculate minimum distance between islands a and b in a circle of n islands"""
    if a == b:
        return 0
    # Clockwise distance
    if b > a:
        clockwise = b - a
    else:
        clockwise = n - a + b
    # Counterclockwise distance
    counterclockwise = n - clockwise
    return min(clockwise, counterclockwise)

def distance_with_closed_bridge(a, b, n, closed):
    """Calculate distance when a specific bridge is closed"""
    if a == b:
        return 0
    
    # Normalize closed bridge (1-indexed to 0-indexed for easier calculation)
    closed = closed - 1
    a = a - 1
    b = b - 1
    
    # Check if we need to cross the closed bridge
    # Bridge i connects island i and (i+1)%n
    
    # Try clockwise path
    clockwise_crosses = False
    if a < b:
        if closed >= a and closed < b:
            clockwise_crosses = True
    else:
        if closed >= a or closed < b:
            clockwise_crosses = True
    
    # Try counterclockwise path
    counterclockwise_crosses = False
    if b < a:
        if closed >= b and closed < a:
            counterclockwise_crosses = True
    else:
        if closed >= b or closed < a:
            counterclockwise_crosses = True
    
    if clockwise_crosses and counterclockwise_crosses:
        # Both paths cross the closed bridge - impossible
        return float('inf')
    elif clockwise_crosses:
        # Must go counterclockwise
        if a < b:
            return n - (b - a)
        else:
            return a - b
    elif counterclockwise_crosses:
        # Must go clockwise
        if a < b:
            return b - a
        else:
            return n - (a - b)
    else:
        # Neither crosses - use shorter path
        if a < b:
            return min(b - a, n - (b - a))
        else:
            return min(a - b, n - (a - b))

# Read input
n, m = map(int, input().split())
x = list(map(int, input().split()))

# Calculate base distances without any closed bridge
base_total = 0
for i in range(m - 1):
    base_total += min_distance(x[i], x[i + 1], n)

# Try closing each bridge and find minimum total distance
min_total = float('inf')

for closed_bridge in range(1, n + 1):
    total = 0
    valid = True
    
    for i in range(m - 1):
        dist = distance_with_closed_bridge(x[i], x[i + 1], n, closed_bridge)
        if dist == float('inf'):
            valid = False
            break
        total += dist
    
    if valid:
        min_total = min(min_total, total)

print(min_total)