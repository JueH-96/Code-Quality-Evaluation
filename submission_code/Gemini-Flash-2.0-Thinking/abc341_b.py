def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = []
    t = []
    for _ in range(n - 1):
        si, ti = map(int, input().split())
        s.append(si)
        t.append(ti)

    while True:
        changed = False
        new_a = list(a)
        for i in range(n - 1):
            if new_a[i] >= s[i]:
                num_exchanges = new_a[i] // s[i]
                if num_exchanges > 0:
                    amount_to_pay = num_exchanges * s[i]
                    amount_to_gain = num_exchanges * t[i]
                    new_a[i] -= amount_to_pay
                    new_a[i+1] += amount_to_gain
                    changed = True
        a = new_a
        if not changed:
            break

    print(a[n-1])

solve()