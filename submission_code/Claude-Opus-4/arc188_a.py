def is_good(s):
    # A string is good if it can be reduced to empty
    # This happens when:
    # 1. Length is even
    # 2. Count of each character has same parity (all even or all odd)
    
    if len(s) == 0:
        return False
    
    count_a = s.count('A')
    count_b = s.count('B')
    count_c = s.count('C')
    
    # All counts must have the same parity
    if count_a % 2 == count_b % 2 == count_c % 2:
        return True
    return False

def count_good_substrings(s):
    n = len(s)
    count = 0
    
    # Check all substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            if is_good(s[i:j]):
                count += 1
    
    return count

def solve():
    n, k = map(int, input().split())
    s = input().strip()
    
    MOD = 998244353
    
    # Find positions of '?'
    question_positions = []
    for i in range(n):
        if s[i] == '?':
            question_positions.append(i)
    
    num_questions = len(question_positions)
    
    if num_questions == 0:
        # No '?' to replace
        if count_good_substrings(s) >= k:
            print(1)
        else:
            print(0)
        return
    
    # Try all possible replacements
    total_ways = 0
    
    # Iterate through all 3^num_questions possibilities
    for mask in range(3 ** num_questions):
        # Create a copy of the string
        chars = list(s)
        
        # Replace each '?' based on mask
        temp_mask = mask
        for i in range(num_questions):
            choice = temp_mask % 3
            temp_mask //= 3
            
            if choice == 0:
                chars[question_positions[i]] = 'A'
            elif choice == 1:
                chars[question_positions[i]] = 'B'
            else:
                chars[question_positions[i]] = 'C'
        
        # Check if this string has at least k good substrings
        current_string = ''.join(chars)
        if count_good_substrings(current_string) >= k:
            total_ways = (total_ways + 1) % MOD
    
    print(total_ways)

solve()