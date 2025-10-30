import sys

def main():
    n = int(sys.stdin.readline())
    lr = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    sum_L = sum(l for l, r in lr)
    sum_R = sum(r for l, r in lr)
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
    
    D = -sum_L
    X = [l for l, r in lr]
    
    for i in range(n):
        l, r = lr[i]
        possible = r - l
        take = min(possible, D)
        X[i] += take
        D -= take
        if D == 0:
            break
    
    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()