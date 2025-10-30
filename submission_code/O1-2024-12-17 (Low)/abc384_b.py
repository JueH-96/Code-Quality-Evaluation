def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    R = int(data[1])
    
    idx = 2
    current_rating = R
    
    for _ in range(N):
        D_i = int(data[idx])
        A_i = int(data[idx+1])
        idx += 2
        
        if D_i == 1:
            # Div.1 rating update condition
            if 1600 <= current_rating <= 2799:
                current_rating += A_i
        else:
            # Div.2 rating update condition
            if 1200 <= current_rating <= 2399:
                current_rating += A_i
    
    print(current_rating)

if __name__ == "__main__":
    main()