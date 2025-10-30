import sys

def main():
    n = int(sys.stdin.readline())
    color_min = {}
    for _ in range(n):
        a, c = map(int, sys.stdin.readline().split())
        if c in color_min:
            if a < color_min[c]:
                color_min[c] = a
        else:
            color_min[c] = a
    print(max(color_min.values()))

if __name__ == "__main__":
    main()