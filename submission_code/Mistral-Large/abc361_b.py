import sys

def cuboids_intersect(a, b, c, d, e, f, g, h, i, j, k, l):
    # Check if the cuboids intersect in the x, y, and z dimensions
    if not (max(a, g) < min(d, j) and max(b, h) < min(e, k) and max(c, i) < min(f, l)):
        return "No"
    return "Yes"

def main():
    input = sys.stdin.read
    data = input().split()

    a, b, c, d, e, f = map(int, data[:6])
    g, h, i, j, k, l = map(int, data[6:])

    result = cuboids_intersect(a, b, c, d, e, f, g, h, i, j, k, l)
    print(result)

if __name__ == "__main__":
    main()