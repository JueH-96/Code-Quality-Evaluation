import bisect
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+m]))
    
    for i in range(1, n+1):
        pos = bisect.bisect_left(A, i)
        if pos < m and A[pos] == i:
            print(0)
        else:
            print(A[pos] - i)

if __name__ == "__main__":
    main()