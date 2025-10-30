# YOUR CODE HERE

import sys

def main():
    N = int(sys.stdin.readline().strip())
    rectangles = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]

    min_x = min(x[0] for x in rectangles)
    max_x = max(x[1] for x in rectangles)
    min_y = min(x[2] for x in rectangles)
    max_y = max(x[3] for x in rectangles)

    S = (max_x - min_x) * (max_y - min_y)

    print(S)

if __name__ == "__main__":
    main()