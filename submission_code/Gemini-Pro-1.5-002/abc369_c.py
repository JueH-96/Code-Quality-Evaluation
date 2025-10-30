# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    for l in range(n):
        for r in range(l, n):
            subsequence = a[l:r+1]
            is_ap = True
            if len(subsequence) > 1:
                diff = subsequence[1] - subsequence[0]
                for i in range(2, len(subsequence)):
                    if subsequence[i] - subsequence[i-1] != diff:
                        is_ap = False
                        break
            if is_ap:
                count += 1
    print(count)

solve()