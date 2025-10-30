def is_good_string(s):
    from collections import Counter
    
    count = Counter(s)
    
    # Check for pairs of identical characters
    pairs = sum(v // 2 for v in count.values())
    
    # Check for triplets of A, B, C
    triplets = min(count['A'], count['B'], count['C'])
    
    return pairs, triplets

def count_good_substrings(s):
    n = len(s)
    good_count = 0
    
    for start in range(n):
        for end in range(start + 1, n + 1):
            substring = s[start:end]
            pairs, triplets = is_good_string(substring)
            if pairs > 0 or triplets > 0:
                good_count += 1
                
    return good_count

def replace_and_count(s, k):
    from itertools import product
    
    n = len(s)
    total_count = 0
    mod = 998244353
    
    # Generate all possible replacements for '?'
    question_marks = s.count('?')
    for replacement in product('ABC', repeat=question_marks):
        temp_s = list(s)
        replacement_index = 0
        
        for i in range(n):
            if temp_s[i] == '?':
                temp_s[i] = replacement[replacement_index]
                replacement_index += 1
        
        good_substring_count = count_good_substrings(''.join(temp_s))
        
        if good_substring_count >= k:
            total_count += 1
            total_count %= mod
            
    return total_count

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    n, k = map(int, data[0].split())
    s = data[1].strip()
    
    result = replace_and_count(s, k)
    print(result)

if __name__ == "__main__":
    main()