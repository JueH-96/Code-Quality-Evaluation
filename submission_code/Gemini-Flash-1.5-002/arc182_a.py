# YOUR CODE HERE
MOD = 998244353

def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    count = 0
    for i in range(2**q):
        s = [0] * n
        valid = True
        for j in range(q):
            p, v = queries[j]
            choice = (i >> j) & 1
            
            if choice == 0:
                cry = False
                for k in range(p):
                    if s[k] > v:
                        cry = True
                        break
                if cry:
                    valid = False
                    break
                for k in range(p):
                    s[k] = v
            else:
                cry = False
                for k in range(p - 1, n):
                    if s[k] > v:
                        cry = True
                        break
                if cry:
                    valid = False
                    break
                for k in range(p - 1, n):
                    s[k] = v

        if valid:
            count += 1

    print(count % MOD)

solve()