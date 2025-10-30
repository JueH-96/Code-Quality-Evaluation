def solve():
    n = int(input())
    s_list = [input() for _ in range(n)]
    canonical_forms = set()
    for s in s_list:
        s_rev = s[::-1]
        if s <= s_rev:
            canonical_form = s
        else:
            canonical_form = s_rev
        canonical_forms.add(canonical_form)
    print(len(canonical_forms))

if __name__ == '__main__':
    solve()