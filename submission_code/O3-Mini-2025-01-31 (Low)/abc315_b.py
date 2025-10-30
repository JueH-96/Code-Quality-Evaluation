def main():
    import sys
    input = sys.stdin.readline
    
    M = int(input().strip())
    D = list(map(int, input().split()))
    
    total = sum(D)
    middle = (total + 1) // 2
    
    cumulative = 0
    for i in range(M):
        cumulative += D[i]
        if cumulative >= middle:
            # The month number is i+1, and the day in that month is:
            # day = middle - (cumulative - D[i])
            day_in_month = middle - (cumulative - D[i])
            print(i + 1, day_in_month)
            return

if __name__ == '__main__':
    main()