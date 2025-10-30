def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    def longest_common_prefix(s1, s2):
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return i

    for k in range(1, n + 1):
        t = s[k - 1]
        min_cost = len(t)  # Cost to make T empty

        for i in range(k - 1):
            prev_s = s[i]
            lcp = longest_common_prefix(t, prev_s)
            cost = (len(t) - lcp) + (len(prev_s) - lcp)
            min_cost = min(min_cost, cost)

        print(min_cost)

solve()