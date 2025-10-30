import bisect
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+M]))
    
    for i in range(1, N+1):
        pos = bisect.bisect_left(A, i)
        print(A[pos] - i)

if __name__ == "__main__":
    main()