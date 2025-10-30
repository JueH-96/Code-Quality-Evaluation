# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    a, b, c, d, e, f = map(int, input[:6])
    g, h, i, j, k, l = map(int, input[6:])

    # Calculate the intersection dimensions
    x1 = max(a, g)
    x2 = min(d, j)
    y1 = max(b, h)
    y2 = min(e, k)
    z1 = max(c, i)
    z2 = min(f, l)

    # Check if there is a positive volume intersection
    if x1 < x2 and y1 < y2 and z1 < z2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()