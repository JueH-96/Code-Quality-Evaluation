import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    Q = list(map(int, data[1:1+n]))
    A = list(map(int, data[1+n:1+2*n]))
    B = list(map(int, data[1+2*n:1+3*n]))
    
    x_max = 10**18
    for i in range(n):
        if A[i] > 0:
            x_candidate = Q[i] // A[i]
            if x_candidate < x_max:
                x_max = x_candidate
                
    best = 0
    if x_max < 0:
        print(0)
        return
        
    for x in range(0, x_max + 1):
        y_max = 10**18
        for i in range(n):
            if B[i] == 0:
                continue
            rem = Q[i] - A[i] * x
            candidate = rem // B[i]
            if candidate < y_max:
                y_max = candidate
        total = x + y_max
        if total > best:
            best = total
            
    print(best)

if __name__ == "__main__":
    main()