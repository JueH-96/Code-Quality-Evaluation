def main():
    import sys
    input = sys.stdin.readline

    N, R = map(int, input().split())
    rating = R
    for _ in range(N):
        d, a = map(int, input().split())
        if d == 1:
            # Div.1: update if 1600 <= rating <= 2799
            if 1600 <= rating <= 2799:
                rating += a
        else:
            # Div.2: update if 1200 <= rating <= 2399
            if 1200 <= rating <= 2399:
                rating += a
    print(rating)

if __name__ == "__main__":
    main()