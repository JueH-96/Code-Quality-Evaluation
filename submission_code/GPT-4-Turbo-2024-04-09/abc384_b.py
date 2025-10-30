def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    R = int(data[1])
    
    contests = []
    for i in range(N):
        D = int(data[2 + 2*i])
        A = int(data[3 + 2*i])
        contests.append((D, A))
    
    current_rating = R
    
    for D, A in contests:
        if D == 1:
            if 1600 <= current_rating <= 2799:
                current_rating += A
        elif D == 2:
            if 1200 <= current_rating <= 2399:
                current_rating += A
    
    print(current_rating)

if __name__ == "__main__":
    main()