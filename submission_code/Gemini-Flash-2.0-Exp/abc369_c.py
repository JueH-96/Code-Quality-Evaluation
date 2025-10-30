def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for l in range(n):
        for r in range(l, n):
            subsequence = a[l:r+1]
            if len(subsequence) <= 1:
                count += 1
            else:
                diff = subsequence[1] - subsequence[0]
                is_arithmetic = True
                for i in range(1, len(subsequence)):
                    if subsequence[i] - subsequence[i-1] != diff:
                        is_arithmetic = False
                        break
                if is_arithmetic:
                    count += 1
    print(count)

solve()