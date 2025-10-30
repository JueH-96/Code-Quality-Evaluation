def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = []
    X = []
    for i in range(N):
        W.append(int(data[1 + 2*i]))
        X.append(int(data[2 + 2*i]))
    
    max_employees = 0
    
    for start_utc in range(24):
        total = 0
        for i in range(N):
            local_time = (start_utc + X[i]) % 24
            if 9 <= local_time < 18:
                total += W[i]
        if total > max_employees:
            max_employees = total
    
    print(max_employees)

if __name__ == "__main__":
    main()