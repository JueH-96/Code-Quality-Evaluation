# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    
    if M == 2:
        if N ==1:
            if A[0] == B[0]:
                print(0)
            else:
                print(1)
        else:
            if A == B:
                print(0)
            else:
                print(-1)
    else:
        total = 0
        for a,b in zip(A,B):
            diff1 = (a - b) % M
            diff2 = (b - a) % M
            total += min(diff1, diff2)
        print(total)

if __name__ == "__main__":
    main()