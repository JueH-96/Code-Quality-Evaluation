import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    A.sort()
    B.sort()
    
    total = 0
    j = 0
    for b in B:
        while j < len(A) and A[j] < b:
            j += 1
        if j >= len(A):
            print(-1)
            return
        total += A[j]
        j += 1
    print(total)

if __name__ == "__main__":
    main()