import math

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    for m in range(1, n + 1):
        subsequences = []
        for i in range(1, 1 << m):
            subsequence = []
            for j in range(m):
                if (i >> j) & 1:
                    subsequence.append(a[j])
            subsequences.append(subsequence)
        
        total_score = 0
        for subsequence in subsequences:
            score = 0
            if len(subsequence) > 1:
                for i in range(len(subsequence) - 1):
                    score += gcd(subsequence[i], subsequence[i+1])
            total_score = (total_score + score) % 998244353
        print(total_score)

solve()