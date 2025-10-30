import sys

def main():
    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:  # no input
        return
    N = data[0]
    heights = data[1:]
    if N == 0 or not heights:
        print(-1)
        return
    first_height = heights[0]
    for idx in range(1, N):
        if heights[idx] > first_height:
            print(idx + 1)  # convert to 1-based index
            return
    print(-1)

if __name__ == "__main__":
    main()