def main():
    N = int(input())
    D = list(map(int,input().split()))

    res = 0
    for i in range(1,11):
        for j in range(1,min(10,D[i-1])+1):
            if str(i) == str(j) and len(str(i)) == len(str(j)):
                res += 1
    # finish at i = 10, so increasing res for each 11, 22, ..., 99
    for i in range(1, N+1):
        if 11 <= D[i-1] <= 99 and str(D[i-1]) == str(D[i-1])[0]*2:
            res += 1
    if N >= 11 and str(N) == '1'* len(str(N)):
        res += 1
    print(res)

if __name__ == '__main__':
    main()