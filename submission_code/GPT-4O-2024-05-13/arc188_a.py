# YOUR CODE HERE
import itertools

MOD = 998244353

def is_good_string(s):
    count_A = s.count('A')
    count_B = s.count('B')
    count_C = s.count('C')
    
    # Check if the string can be reduced to empty by the given operations
    while count_A > 0 or count_B > 0 or count_C > 0:
        if count_A >= 2:
            count_A -= 2
        elif count_B >= 2:
            count_B -= 2
        elif count_C >= 2:
            count_C -= 2
        elif count_A >= 1 and count_B >= 1 and count_C >= 1:
            count_A -= 1
            count_B -= 1
            count_C -= 1
        else:
            return False
    return True

def count_good_substrings(s):
    n = len(s)
    count = 0
    for length in range(1, n + 1):
        for start in range(n - length + 1):
            if is_good_string(s[start:start + length]):
                count += 1
    return count

def solve(N, K, S):
    question_marks = S.count('?')
    possible_replacements = itertools.product('ABC', repeat=question_marks)
    valid_count = 0
    
    for replacement in possible_replacements:
        new_string = list(S)
        replacement_index = 0
        for i in range(N):
            if new_string[i] == '?':
                new_string[i] = replacement[replacement_index]
                replacement_index += 1
        new_string = ''.join(new_string)
        if count_good_substrings(new_string) >= K:
            valid_count += 1
            valid_count %= MOD
    
    print(valid_count)

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    solve(N, K, S)