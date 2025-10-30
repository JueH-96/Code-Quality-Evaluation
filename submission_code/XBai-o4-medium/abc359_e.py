def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    res = []
    current = 0
    depth = 1
    prev = 0
    for h in H:
        if h > prev:
            current += (h - prev) * depth
            depth *= 2
        else:
            current += h * 1
            depth = 1
        prev = h
        res.append(current)
    
    # Adjusting the result to match the required output
    # Each answer is current + 1 for the first element, but for subsequent elements, it's cumulative
    # However, the correct adjustment is to add 1 to each element
    adjusted = [x + 1 for x in res]
    print(' '.join(map(str, adjusted)))

if __name__ == '__main__':
    main()