def solve():
    n = int(input())
    s_list = []
    for _ in range(n):
        s_list.append(input())
    
    distinct_sticks_canonical_forms = set()
    for s in s_list:
        reversed_s = s[::-1]
        canonical_form = min(s, reversed_s)
        distinct_sticks_canonical_forms.add(canonical_form)
        
    print(len(distinct_sticks_canonical_forms))

if __name__ == '__main__':
    solve()