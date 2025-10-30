def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    color_min = {}
    for _ in range(n):
        a = int(input[idx])
        c = int(input[idx + 1])
        idx += 2
        if c in color_min:
            if a < color_min[c]:
                color_min[c] = a
        else:
            color_min[c] = a
    print(max(color_min.values()))

if __name__ == "__main__":
    main()