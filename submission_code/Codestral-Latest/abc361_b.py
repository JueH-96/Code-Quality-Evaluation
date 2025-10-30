# YOUR CODE HERE
import sys

def read_input():
    return [list(map(int, input().split())) for _ in range(2)]

def check_intersection(cuboid1, cuboid2):
    a, b, c, d, e, f = cuboid1
    g, h, i, j, k, l = cuboid2

    # Check for overlap in each dimension
    overlap_x = max(0, min(d, j) - max(a, g))
    overlap_y = max(0, min(e, k) - max(b, h))
    overlap_z = max(0, min(f, l) - max(c, i))

    # If there is overlap in all dimensions, the volume is positive
    return overlap_x > 0 and overlap_y > 0 and overlap_z > 0

def main():
    cuboid1, cuboid2 = read_input()
    if check_intersection(cuboid1, cuboid2):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()