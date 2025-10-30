import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    for m in range(1, n + 1):
        total_score = 0
        for i in range(1, 1 << m):
            subsequence = []
            for j in range(m):
                if (i >> j) & 1:
                    subsequence.append(a[j])
            
            score = 0
            for k in range(len(subsequence) - 1):
                score += gcd(subsequence[k], subsequence[k+1])
            total_score = (total_score + score) % 998244353
        print(total_score)

solve()