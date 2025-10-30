import sys

def main():
    n, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    A.sort(reverse=True)
    B.sort(reverse=True)
    prefixA = [0] * (n + 1)
    prefixB = [0] * (n + 1)
    for i in range(n):
        prefixA[i+1] = prefixA[i] + A[i]
        prefixB[i+1] = prefixB[i] + B[i]
    for k in range(1, n+1):
        if prefixA[k] > X or prefixB[k] > Y:
            print(k)
            return
    print(n)

if __name__ == '__main__':
    main()