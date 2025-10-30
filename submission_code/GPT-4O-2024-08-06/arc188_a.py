def is_good_string(s):
    count_a = s.count('A')
    count_b = s.count('B')
    count_c = s.count('C')
    return (count_a % 2 == 0 and count_b % 2 == 0 and count_c % 2 == 0 and
            count_a >= count_b and count_a >= count_c)

def count_good_substrings(s):
    n = len(s)
    good_count = 0
    for start in range(n):
        for end in range(start + 1, n + 1):
            if is_good_string(s[start:end]):
                good_count += 1
    return good_count

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    MOD = 998244353
    
    from itertools import product
    
    question_marks = S.count('?')
    total_ways = 0
    
    for replacement in product('ABC', repeat=question_marks):
        new_string = list(S)
        replacement_index = 0
        for i in range(N):
            if new_string[i] == '?':
                new_string[i] = replacement[replacement_index]
                replacement_index += 1
        new_string = ''.join(new_string)
        
        if count_good_substrings(new_string) >= K:
            total_ways += 1
            total_ways %= MOD
    
    print(total_ways)