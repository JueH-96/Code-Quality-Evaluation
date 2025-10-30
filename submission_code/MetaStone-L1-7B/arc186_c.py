import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        M = int(data[idx+1])
        idx += 2
        total = 0
        for _ in range(N):
            V = int(data[idx])
            P = int(data[idx+1])
            idx += 2
            if V > P:
                total += (V - P)
        print(total)

if __name__ == "__main__":
    main()