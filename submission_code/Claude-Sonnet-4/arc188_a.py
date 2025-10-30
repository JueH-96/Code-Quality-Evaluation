def is_good(s):
    if not s:
        return True
    
    count_a = s.count('A')
    count_b = s.count('B') 
    count_c = s.count('C')
    
    # Remove maximum possible ABC triplets
    min_abc = min(count_a, count_b, count_c)
    count_a -= min_abc
    count_b -= min_abc
    count_c -= min_abc
    
    # Check if remaining counts are all even
    return count_a % 2 == 0 and count_b % 2 == 0 and count_c % 2 == 0

N, K = map(int, input().split())
S = input().strip()

MOD = 998244353

# Find positions of '?'
question_positions = []
for i in range(N):
    if S[i] == '?':
        question_positions.append(i)

num_questions = len(question_positions)

if num_questions == 0:
    # No question marks, just check if current string has >= K good substrings
    good_count = 0
    for i in range(N):
        for j in range(i, N):
            substring = S[i:j+1]
            if is_good(substring):
                good_count += 1
    
    if good_count >= K:
        print(1)
    else:
        print(0)
else:
    valid_ways = 0
    
    # Try all possible replacements
    for mask in range(3 ** num_questions):
        # Create the string with replacements
        current_string = list(S)
        temp_mask = mask
        
        for pos in question_positions:
            replacement = temp_mask % 3
            if replacement == 0:
                current_string[pos] = 'A'
            elif replacement == 1:
                current_string[pos] = 'B'
            else:
                current_string[pos] = 'C'
            temp_mask //= 3
        
        current_string = ''.join(current_string)
        
        # Count good substrings
        good_count = 0
        for i in range(N):
            for j in range(i, N):
                substring = current_string[i:j+1]
                if is_good(substring):
                    good_count += 1
        
        if good_count >= K:
            valid_ways += 1
    
    print(valid_ways % MOD)