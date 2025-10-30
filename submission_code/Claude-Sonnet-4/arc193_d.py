def solve():
    n = int(input())
    a = input().strip()
    b = input().strip()
    
    # Get piece and target positions
    pieces = [i for i in range(n) if a[i] == '1']
    targets = [i for i in range(n) if b[i] == '1']
    
    # Check if possible
    if len(pieces) != len(targets):
        return -1
    
    if not pieces:
        return 0
    
    # Sort both
    pieces.sort()
    targets.sort()
    
    # Calculate minimum operations
    # The key insight is that we can assign pieces to targets greedily
    total_ops = 0
    for i in range(len(pieces)):
        total_ops += abs(pieces[i] - targets[i])
    
    return total_ops

t = int(input())
for _ in range(t):
    result = solve()
    print(result)