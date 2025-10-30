def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    rating = int(next(it))
    for _ in range(N):
        D = int(next(it))
        A = int(next(it))
        if D == 1:
            # ARC Div.1: rating update if current rating is between 1600 and 2799, inclusive.
            if 1600 <= rating <= 2799:
                rating += A
        elif D == 2:
            # ARC Div.2: rating update if current rating is between 1200 and 2399, inclusive.
            if 1200 <= rating <= 2399:
                rating += A
    sys.stdout.write(str(rating))

if __name__ == '__main__':
    main()