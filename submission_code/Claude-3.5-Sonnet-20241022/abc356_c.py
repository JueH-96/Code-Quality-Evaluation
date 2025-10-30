def check_combination(n, m, k, tests, real_keys):
    for test in tests:
        count = len([1 for key in test[:-1] if real_keys[key-1]])
        if test[-1] == 'o' and count < k:
            return False
        if test[-1] == 'x' and count >= k:
            return False
    return True

def solve():
    n, m, k = map(int, input().split())
    tests = []
    for _ in range(m):
        test = list(map(int, input().split()[:-1])) + [input().split()[-1]]
        tests.append(test)
    
    ans = 0
    for state in range(1 << n):
        real_keys = [(state >> i) & 1 for i in range(n)]
        if check_combination(n, m, k, tests, real_keys):
            ans += 1
            
    print(ans)

solve()