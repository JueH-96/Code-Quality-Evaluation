# YOUR CODE HERE
def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    M, D = map(int, data[:2])
    y, m, d = map(int, data[2:])
    
    if d < D:
        # Just move to the next day
        d += 1
    else:
        # Wrap to the next month or next year if needed
        d = 1
        if m < M:
            m += 1
        else:
            m = 1
            y += 1
    
    print(y, m, d)

main()