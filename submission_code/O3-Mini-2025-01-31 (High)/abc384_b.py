def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    rating = int(input_data[1])
    index = 2
    for _ in range(N):
        d = int(input_data[index])
        a = int(input_data[index+1])
        index += 2
        if d == 1:
            # ARC Div. 1: eligible if rating is between 1600 and 2799 inclusive
            if 1600 <= rating <= 2799:
                rating += a
        elif d == 2:
            # ARC Div. 2: eligible if rating is between 1200 and 2399 inclusive
            if 1200 <= rating <= 2399:
                rating += a
    sys.stdout.write(str(rating))

if __name__ == '__main__':
    main()