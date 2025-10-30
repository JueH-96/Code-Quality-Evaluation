def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    current_score_sum = 0
    results = []
    mod = 998244353
    for m in range(1, n + 1):
        if m == 1:
            current_score_sum = 0
        else:
            next_score_sum = (2 * current_score_sum) % mod
            term_sum = 0
            power_of_2 = 1
            for i in range(1, m):
                g = gcd(a[i-1], a[m-1])
                term = (g * power_of_2) % mod
                term_sum = (term_sum + term) % mod
                power_of_2 = (power_of_2 * 2) % mod
            current_score_sum = (next_score_sum + term_sum) % mod
        results.append(current_score_sum)
    for score in results:
        print(score)

if __name__ == '__main__':
    solve()