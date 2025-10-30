import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+n+n-1]))
    
    A.sort()
    B.sort()
    
    pre = [0] * (n+1)
    pre[0] = 0
    j = 0
    for i in range(n):
        while j < len(B) and B[j] < A[i]:
            j += 1
        if j < len(B):
            pre[i+1] = j+1
            j += 1
        else:
            pre[i+1] = 10**18
    
    F = []
    if n >= 2:
        for i in range(len(B)):
            F.append(B[i] - A[i+1])
    
    sufF = [0] * n
    if n >= 2:
        sufF[len(B)-1] = F[len(B)-1]
        for i in range(len(B)-2, -1, -1):
            sufF[i] = min(F[i], sufF[i+1])
    
    ans = 10**18
    for i in range(n):
        if pre[i] > i:
            continue
        m = n - i - 1
        if m == 0:
            if A[i] < ans:
                ans = A[i]
        else:
            if sufF[i] >= 0:
                if A[i] < ans:
                    ans = A[i]
    
    if ans == 10**18:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()