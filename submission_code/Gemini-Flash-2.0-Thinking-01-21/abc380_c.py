def solve():
    n, k = map(int, input().split())
    s = input()

    blocks = []
    i = 0
    while i < n:
        if s[i] == '1':
            start = i
            while i < n and s[i] == '1':
                i += 1
            end = i - 1
            blocks.append((start, end))
        else:
            i += 1

    l_k_minus_1, r_k_minus_1 = blocks[k - 2]
    l_k, r_k = blocks[k - 1]

    ans = ""
    ans += s[:r_k_minus_1 + 1]
    ans += s[l_k:r_k + 1]
    ans += s[r_k_minus_1 + 1:l_k]
    ans += s[r_k + 1:]

    print(ans)

solve()