def solve():
    n = int(input())
    a = list(map(int, input().split()))

    count = 0
    for l in range(n):
        for r in range(l, n):
            subsequence = a[l:r+1]
            length = len(subsequence)

            if length <= 2:
                count += 1
            else:
                diff = subsequence[1] - subsequence[0]
                is_ap = True
                for i in range(2, length):
                    if subsequence[i] - subsequence[i-1] != diff:
                        is_ap = False
                        break
                if is_ap:
                    count += 1
    print(count)

solve()