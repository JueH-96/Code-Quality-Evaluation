# YOUR CODE HERE
s = input()
n = int(input())

def solve():
    q_count = s.count('?')
    max_val = -1

    for i in range(1 << q_count):
        temp_s = list(s)
        binary = bin(i)[2:].zfill(q_count)
        k = 0
        for j in range(len(temp_s)):
            if temp_s[j] == '?':
                temp_s[j] = binary[k]
                k += 1
        
        cur_val = int("".join(temp_s), 2)
        if cur_val <= n:
            max_val = max(max_val, cur_val)

    print(max_val)

solve()