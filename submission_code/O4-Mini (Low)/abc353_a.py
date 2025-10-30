def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    heights = [int(next(it)) for _ in range(n)]
    first_height = heights[0]
    answer = -1
    for idx in range(1, n):
        if heights[idx] > first_height:
            answer = idx + 1  # convert to 1-based index
            break
    print(answer)

if __name__ == "__main__":
    main()