def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # We'll record the minimum deliciousness for each color
    color_min = {}
    for _ in range(N):
        deliciousness = int(next(it))
        color = int(next(it))
        # Update the dictionary: for each color, store the smallest deliciousness seen.
        if color in color_min:
            if deliciousness < color_min[color]:
                color_min[color] = deliciousness
        else:
            color_min[color] = deliciousness

    # To maximize the worst-case flavor, choose the color with the highest minimum deliciousness.
    result = max(color_min.values()) if color_min else 0
    print(result)

if __name__ == "__main__":
    main()