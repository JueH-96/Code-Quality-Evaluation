def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    powers_of_2 = [1] * n
    for i in range(1, n):
        powers_of_2[i] = (powers_of_2[i-1] * 2) % mod
        
    scores_sum = 0
    results = []
    
    results.append(0)
    
    for m in range(2, n + 1):
        current_t = 0
        for i in range(1, m):
            g = gcd(a[i-1], a[m-1])
            term = (g * powers_of_2[i-1]) % mod
            current_t = (current_t + term) % mod
        scores_sum = (2 * scores_sum + current_t) % mod
        results.append(scores_sum)
        
    for score in results:
        print(score)

if __name__ == '__main__':
    solve()