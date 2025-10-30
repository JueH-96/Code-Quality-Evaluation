import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    A.sort()
    total = 0
    for i in range(N):
        a = A[i]
        for j in range(i+1, N):
            b = A[j]
            if b >= a:
                total += b // a
            else:
                total += a // b
    print(total)

if __name__ == "__main__":
    main()