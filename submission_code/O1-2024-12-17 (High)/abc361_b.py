def main():
    # Read the inputs
    a, b, c, d, e, f = map(int, input().split())
    g, h, i, j, k, l = map(int, input().split())

    # Check for positive overlap in each dimension
    if max(a, g) < min(d, j) and max(b, h) < min(e, k) and max(c, i) < min(f, l):
        print("Yes")
    else:
        print("No")

# Do not forget to call main otherwise you will not be awarded any points.
main()