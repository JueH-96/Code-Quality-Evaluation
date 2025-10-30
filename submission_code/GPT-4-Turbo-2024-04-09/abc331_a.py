import sys
input = sys.stdin.read

def main():
    data = input().split()
    M = int(data[0])
    D = int(data[1])
    y = int(data[2])
    m = int(data[3])
    d = int(data[4])
    
    if d < D:
        d += 1
    elif m < M:
        m += 1
        d = 1
    else:
        y += 1
        m = 1
        d = 1
    
    print(y, m, d)

if __name__ == "__main__":
    main()