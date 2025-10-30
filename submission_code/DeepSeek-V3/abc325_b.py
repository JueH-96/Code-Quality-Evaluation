# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = []
    X = []
    for i in range(N):
        W.append(int(data[2*i+1]))
        X.append(int(data[2*i+2]))
    
    max_employees = 0
    
    for start in range(24):
        total = 0
        for i in range(N):
            local_time = (start + X[i]) % 24
            if 9 <= local_time < 18:
                total += W[i]
        if total > max_employees:
            max_employees = total
    
    print(max_employees)

if __name__ == "__main__":
    main()