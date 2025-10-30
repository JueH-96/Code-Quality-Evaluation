import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    T = int(data[1])
    P = int(data[2])
    L = list(map(int, data[3:3+n]))
    
    req_days = [max(0, T - l) for l in L]
    req_days.sort()
    print(req_days[P-1])

if __name__ == "__main__":
    main()