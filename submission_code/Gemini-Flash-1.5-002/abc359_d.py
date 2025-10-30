# YOUR CODE HERE
MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def solve():
    n, k = map(int, input().split())
    s = input()
    
    q_count = s.count('?')
    
    count = 0
    for i in range(2**q_count):
        temp = bin(i)[2:].zfill(q_count)
        current_string = list(s)
        temp_index = 0
        for j in range(n):
            if current_string[j] == '?':
                current_string[j] = 'A' if temp[temp_index] == '0' else 'B'
                temp_index += 1
        
        current_string = "".join(current_string)
        is_good = True
        for j in range(n - k + 1):
            substring = current_string[j:j+k]
            if is_palindrome(substring):
                is_good = False
                break
        if is_good:
            count += 1
    print(count % MOD)

solve()