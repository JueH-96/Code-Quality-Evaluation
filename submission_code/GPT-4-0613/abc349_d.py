def solve(L, R):
    ans = []
    while L < R:
        for i in range(60, -1, -1):
            if (L >> i << i) == L and (L + (1 << i)) <= R:
                ans.append((L, L + (1 << i)))
                L += 1 << i
                break
    print(len(ans))
    for l, r in ans:
        print(l, r)

L, R = map(int, input().split())
solve(L, R)