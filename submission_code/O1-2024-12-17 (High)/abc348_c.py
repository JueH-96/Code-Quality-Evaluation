def main():
    import sys
    data = sys.stdin.read().strip().split()
    n = int(data[0])

    # Dictionary to store the minimum deliciousness per color
    color_min = {}

    idx = 1
    for _ in range(n):
        a = int(data[idx])
        c = int(data[idx + 1])
        idx += 2

        if c not in color_min:
            color_min[c] = a
        else:
            # Update the minimum deliciousness for this color
            if a < color_min[c]:
                color_min[c] = a

    # We want the maximum of these minimum values
    print(max(color_min.values()))

# Do not forget to call main()
main()