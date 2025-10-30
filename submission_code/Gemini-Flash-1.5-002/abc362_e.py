# YOUR CODE HERE
MOD = 998244353

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    ans = []
    for k in range(1, n + 1):
        count = 0
        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(a[j])
            
            if len(subsequence) == k:
                is_arithmetic = True
                if k > 1:
                    diff = subsequence[1] - subsequence[0]
                    for l in range(2, k):
                        if subsequence[l] - subsequence[l-1] != diff:
                            is_arithmetic = False
                            break
                if is_arithmetic:
                    count += 1
        ans.append(count)
    print(*ans)

solve()