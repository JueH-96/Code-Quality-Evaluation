def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    total_A = sum(A)
    total_B = sum(B)
    
    # If even eating all dishes, neither threshold is exceeded,
    # then he will end up eating all dishes.
    if total_A <= X and total_B <= Y:
        sys.stdout.write(str(N))
        return
    
    ans = float('inf')
    
    # Option 1: try to exceed the sweetness threshold.
    if total_A > X:
        # Sort dishes by sweetness in descending order.
        A_sorted = sorted(A, reverse=True)
        cumulative = 0
        count = 0
        for sweet in A_sorted:
            cumulative += sweet
            count += 1
            if cumulative > X:
                ans = min(ans, count)
                break
    
    # Option 2: try to exceed the saltiness threshold.
    if total_B > Y:
        # Sort dishes by saltiness in descending order.
        B_sorted = sorted(B, reverse=True)
        cumulative = 0
        count = 0
        for salty in B_sorted:
            cumulative += salty
            count += 1
            if cumulative > Y:
                ans = min(ans, count)
                break
                
    # If for some reason no ordering can trigger any threshold,
    # then he eats all dishes.
    if ans == float('inf'):
        ans = N
        
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()