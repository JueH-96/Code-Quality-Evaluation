import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+2*n]))
    
    A.sort(reverse=True)
    B.sort(reverse=True)
    
    totalA = 0
    totalB = 0
    for i in range(n):
        totalA += A[i]
        totalB += B[i]
        if totalA > X or totalB > Y:
            print(i + 1)
            return
            
    print(n)

if __name__ == "__main__":
    main()