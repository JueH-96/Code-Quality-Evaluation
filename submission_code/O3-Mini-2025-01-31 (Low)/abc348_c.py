def main():
    import sys
    input = sys.stdin.readline
    
    # Read N
    N = int(input().strip())
    
    # Dictionary to hold the minimum deliciousness for each color.
    color_min = {}
    
    for _ in range(N):
        line = input().strip()
        if not line:
            continue
        a, c = map(int, line.split())
        # If color is not in dictionary, set the current deliciousness.
        if c not in color_min:
            color_min[c] = a
        # Otherwise update the minimum deliciousness.
        else:
            color_min[c] = min(color_min[c], a)
    
    # The best we can do by choosing that color is the one with maximum minimum deliciousness.
    answer = max(color_min.values())
    
    print(answer)

if __name__ == '__main__':
    main()