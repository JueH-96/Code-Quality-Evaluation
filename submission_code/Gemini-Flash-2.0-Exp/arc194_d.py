def solve():
    n = int(input())
    s = input()
    
    def is_valid(seq):
        balance = 0
        for char in seq:
            if char == '(':
                balance += 1
            else:
                balance -= 1
            if balance < 0:
                return False
        return balance == 0

    def reverse_substring(s, l, r):
        temp = list(s)
        for i in range(l, r + 1):
            if temp[l + r - i] == '(':
                temp[i] = ')'
            else:
                temp[i] = '('
        return "".join(temp)

    q = {s}
    visited = {s}
    
    while q:
        curr = q.pop()
        
        for i in range(n):
            for j in range(i, n):
                sub = curr[i:j+1]
                if is_valid(sub):
                    new_s = reverse_substring(curr, i, j)
                    if new_s not in visited:
                        q.add(new_s)
                        visited.add(new_s)
    
    print(len(visited) % 998244353)

solve()