def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    # Dictionary to store the minimum deliciousness for each color
    min_by_color = {}
    for _ in range(N):
        a, c = map(int, input().split())
        # If color c seen first time, set to a, else update min
        if c in min_by_color:
            if a < min_by_color[c]:
                min_by_color[c] = a
        else:
            min_by_color[c] = a

    # Among all colors, pick the one whose min deliciousness is largest
    answer = max(min_by_color.values())
    print(answer)

if __name__ == "__main__":
    main()