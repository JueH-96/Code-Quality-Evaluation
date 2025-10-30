def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N, R = map(int, data[:2])
    rating = R
    idx = 2
    
    for _ in range(N):
        D_i, A_i = map(int, data[idx:idx+2])
        idx += 2
        if D_i == 1:
            # Div.1 rating update condition
            if 1600 <= rating <= 2799:
                rating += A_i
        else:
            # Div.2 rating update condition
            if 1200 <= rating <= 2399:
                rating += A_i

    print(rating)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()