def solve():
    n, k = map(int, input().split())
    s = input()
    
    def is_good(t):
        while True:
            op1 = False
            for i in range(len(t) - 1):
                if t[i] == t[i+1]:
                    t = t[:i] + t[i+2:]
                    op1 = True
                    break
            if not op1:
                if 'A' in t and 'B' in t and 'C' in t:
                    t = t.replace('A', '', 1).replace('B', '', 1).replace('C', '', 1)
                else:
                    break
        return len(t) == 0

    def count_good_substrings(t):
        count = 0
        for i in range(len(t)):
            for j in range(i, len(t)):
                if is_good(t[i:j+1]):
                    count += 1
        return count

    q_count = s.count('?')
    ans = 0
    
    for i in range(3**q_count):
        temp = s
        cur = i
        for j in range(q_count):
            digit = cur % 3
            cur //= 3
            temp = temp.replace('?', 'ABC'[digit], 1)
        
        if count_good_substrings(temp) >= k:
            ans += 1

    print(ans % 998244353)

solve()