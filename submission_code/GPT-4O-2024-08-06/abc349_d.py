def solve():
    import sys
    input = sys.stdin.read
    data = input().strip()
    L, R = map(int, data.split())
    
    result = []
    
    current = L
    while current < R:
        # Find the largest power of two range starting from current
        length = 1
        while (current // length) * length != current or current + length > R:
            length *= 2
        
        # Add the good sequence
        result.append((current, current + length))
        
        # Move to the next starting point
        current += length
    
    # Output the result
    print(len(result))
    for l, r in result:
        print(l, r)