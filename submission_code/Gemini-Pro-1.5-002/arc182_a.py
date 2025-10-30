# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    ans = 0
    mod = 998244353

    def check(ops):
        s = [0] * n
        for i in range(q):
            p, v = queries[i]
            op = ops[i]
            if op == 0:
                if any(s[j] > v for j in range(p)):
                    return False
                for j in range(p):
                    s[j] = v
            else:
                if any(s[j] > v for j in range(p - 1, n)):
                    return False
                for j in range(p - 1, n):
                    s[j] = v
        return True

    def generate_ops(k, current_ops):
        nonlocal ans
        if k == q:
            if check(current_ops):
                ans = (ans + 1) % mod
            return

        generate_ops(k + 1, current_ops + [0])
        generate_ops(k + 1, current_ops + [1])

    generate_ops(0, [])
    print(ans)

solve()