# YOUR CODE HERE
MOD = 998244353

def pow_mod(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def inv_mod(x, p):
    return pow_mod(x, p - 2, p)

N = int(input())
matches = []
for i in range(N - 1):
    p, q = map(int, input().split())
    matches.append((p, q))

ans = [0] * N
for i in range(N - 1):
    p, q = matches[i]
    p -= 1
    q -= 1
    team1_size = 1
    team2_size = 1
    team1_members = {p}
    team2_members = {q}
    
    
    for j in range(i):
        prev_p, prev_q = matches[j]
        prev_p -=1
        prev_q -=1
        
        if prev_p in team1_members:
            team1_size +=1
        else:
            team2_size +=1
        if prev_q in team1_members:
            team1_size +=1
        else:
            team2_size +=1
            
        
        
    prob_team1_wins = team1_size * inv_mod(team1_size + team2_size, MOD) % MOD
    
    for member in team1_members:
        ans[member] = (ans[member] + prob_team1_wins) % MOD
    for member in team2_members:
        ans[member] = (ans[member] + (1 - prob_team1_wins + MOD) % MOD) % MOD

print(*ans)