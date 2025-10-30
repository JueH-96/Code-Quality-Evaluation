import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    low = 0
    high = N // 2
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        ok = True
        for i in range(mid):
            if A[i] * 2 > A[N - mid + i]:
                ok = False
                break
        if ok:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == "__main__":
    main()