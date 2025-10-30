# YOUR CODE HERE
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    for i in range(1, N+1):
        idx = bisect.bisect_left(A, i)
        if idx < M and A[idx] == i:
            print(0)
        else:
            if idx < M:
                print(A[idx] - i)
            else:
                print(0)

if __name__ == "__main__":
    main()