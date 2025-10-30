def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    R = int(input_data[1])
    index = 2
    rating = R
    for _ in range(N):
        D = int(input_data[index])
        A = int(input_data[index+1])
        index += 2
        if D == 1:
            # Div. 1: rating update when rating in [1600, 2799]
            if 1600 <= rating <= 2799:
                rating += A
        else:
            # Div. 2: rating update when rating in [1200, 2399]
            if 1200 <= rating <= 2399:
                rating += A
    sys.stdout.write(str(rating))
    
if __name__ == '__main__':
    main()