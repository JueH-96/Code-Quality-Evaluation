import sys

def main():
    A = []
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            A.append(x)
        else:
            k = int(parts[1])
            print(A[len(A) - k])
            
if __name__ == "__main__":
    main()