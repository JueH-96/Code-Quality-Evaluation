def is_good(s):
    n = len(s)
    q = [(list(s), 0)]
    visited = {tuple(list(s))}
    while q:
        curr_s, steps = q.pop(0)
        if not curr_s:
            return True
        
        for i in range(len(curr_s) - 1):
            if curr_s[i] == curr_s[i+1]:
                next_s = curr_s[:i] + curr_s[i+2:]
                if tuple(next_s) not in visited:
                    visited.add(tuple(next_s))
                    q.append((next_s, steps + 1))

        counts = {'A': 0, 'B': 0, 'C': 0}
        for char in curr_s:
            counts[char] += 1
        if counts['A'] > 0 and counts['B'] > 0 and counts['C'] > 0:
            next_s = []
            a_found = False
            b_found = False
            c_found = False
            for char in curr_s:
                if char == 'A' and not a_found:
                    a_found = True
                elif char == 'B' and not b_found:
                    b_found = True
                elif char == 'C' and not c_found:
                    c_found = True
                else:
                    next_s.append(char)
            if tuple(next_s) not in visited:
                visited.add(tuple(next_s))
                q.append((next_s, steps + 1))

    return False

def solve():
    n, k = map(int, input().split())
    s = input()
    
    q_count = s.count('?')
    ans = 0
    
    for i in range(3**q_count):
        temp = i
        curr_s = list(s)
        for j in range(n):
            if curr_s[j] == '?':
                digit = temp % 3
                if digit == 0:
                    curr_s[j] = 'A'
                elif digit == 1:
                    curr_s[j] = 'B'
                else:
                    curr_s[j] = 'C'
                temp //= 3
        
        curr_s = "".join(curr_s)
        good_count = 0
        for j in range(n):
            for l in range(j, n):
                if is_good(curr_s[j:l+1]):
                    good_count += 1
        
        if good_count >= k:
            ans += 1

    print(ans % 998244353)

solve()