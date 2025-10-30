def main():
    import sys

    # Read prices
    R, G, B = map(int, sys.stdin.readline().split())

    # Read disliked color
    C = sys.stdin.readline().strip()

    # Choose the cheapest allowed pen
    if C == "Red":
        ans = min(G, B)
    elif C == "Green":
        ans = min(R, B)
    else:  # C == "Blue"
        ans = min(R, G)

    print(ans)


if __name__ == "__main__":
    main()