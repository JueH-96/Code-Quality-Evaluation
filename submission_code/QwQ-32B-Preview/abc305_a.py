def main():
    N = int(input())
    
    lower = (N // 5) * 5
    upper = ((N + 4) // 5) * 5
    
    diff_lower = abs(N - lower)
    diff_upper = abs(N - upper)
    
    if diff_lower < diff_upper:
        print(lower)
    else:
        print(upper)

if __name__ == "__main__":
    main()