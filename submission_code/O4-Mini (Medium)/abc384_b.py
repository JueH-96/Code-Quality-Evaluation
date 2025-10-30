def main():
    import sys
    input = sys.stdin.readline
    N, R = map(int, input().split())
    rating = R
    for _ in range(N):
        D, A = map(int, input().split())
        if D == 1:
            # Div.1 update if rating in [1600, 2799]
            if 1600 <= rating <= 2799:
                rating += A
        else:
            # Div.2 update if rating in [1200, 2399]
            if 1200 <= rating <= 2399:
                rating += A
    print(rating)

if __name__ == "__main__":
    main()