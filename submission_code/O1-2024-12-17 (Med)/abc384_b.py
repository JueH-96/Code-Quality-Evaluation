def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    R = int(data[1])
    
    rating = R
    idx = 2
    
    for _ in range(N):
        D_i = int(data[idx])
        A_i = int(data[idx+1])
        idx += 2
        
        if D_i == 1:
            # Div. 1: update range [1600, 2799]
            if 1600 <= rating <= 2799:
                rating += A_i
        else:
            # Div. 2: update range [1200, 2399]
            if 1200 <= rating <= 2399:
                rating += A_i
    
    print(rating)

# Call the main function
if __name__ == "__main__":
    main()