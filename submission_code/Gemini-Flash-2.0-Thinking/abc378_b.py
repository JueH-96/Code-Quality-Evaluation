def solve():
    n = int(input())
    garbage_info = []
    for _ in range(n):
        q, r = map(int, input().split())
        garbage_info.append((q, r))

    q_queries = int(input())
    for _ in range(q_queries):
        t, d = map(int, input().split())
        q, r = garbage_info[t - 1]

        if d % q == r:
            print(d)
        else:
            next_d = d
            while True:
                next_d += 1
                if next_d % q == r:
                    print(next_d)
                    break

solve()