import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)

def modular_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    return x % m

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

MOD = 998244353

def solve(N, p, q):
    expected_wins = [0] * N
    team_size = [1] * N
    
    for i in range(N - 1):
        p_i, q_i = p[i] - 1, q[i] - 1
        a, b = team_size[p_i], team_size[q_i]
        
        p_win = a / (a + b)
        q_win = b / (a + b)
        
        expected_wins[p_i] = (expected_wins[p_i] + p_win) % MOD
        expected_wins[q_i] = (expected_wins[q_i] + q_win) % MOD
        
        team_size[p_i] += team_size[q_i]
        team_size[q_i] = team_size[p_i]
    
    for i in range(N):
        x = expected_wins[i]
        y = team_size[i]
        g = gcd(x, MOD)
        x //= g
        y //= g
        expected_wins[i] = pow(y, MOD - 2, MOD) * x % MOD
    
    return expected_wins

if __name__ == '__main__':
    N = int(input())
    p = []
    q = []
    for _ in range(N - 1):
        pi, qi = map(int, input().split())
        p.append(pi)
        q.append(qi)
    
    result = solve(N, p, q)
    print(' '.join(map(str, result)))