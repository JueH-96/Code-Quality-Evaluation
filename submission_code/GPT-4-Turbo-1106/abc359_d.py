from itertools import product

MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    question_marks = S.count('?')
    if question_marks == 0:
        # If there are no question marks, check if the string is already good
        return int(all(not is_palindrome(S[i:i+K]) for i in range(N-K+1)))
    
    # Generate all possible combinations for the question marks
    combinations = product('AB', repeat=question_marks)
    good_strings_count = 0
    
    for combination in combinations:
        temp_string = list(S)
        combination_index = 0
        
        # Replace question marks with the current combination
        for i in range(N):
            if temp_string[i] == '?':
                temp_string[i] = combination[combination_index]
                combination_index += 1
        
        # Check if the generated string is good
        if all(not is_palindrome(temp_string[i:i+K]) for i in range(N-K+1)):
            good_strings_count += 1
    
    return good_strings_count % MOD

# Read input
N, K = map(int, input().split())
S = input().strip()

# Calculate and print the answer
print(count_good_strings(N, K, S))