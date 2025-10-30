from itertools import permutations

def is_palindrome(s):
    return s == s[::-1]

def count_non_palindromic_permutations(N, K, S):
    unique_permutations = set(permutations(S))
    count = 0
    
    for perm in unique_permutations:
        perm_str = ''.join(perm)
        found_palindrome = False
        
        for i in range(N - K + 1):
            if is_palindrome(perm_str[i:i + K]):
                found_palindrome = True
                break
        
        if not found_palindrome:
            count += 1
            
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, K = map(int, data[0].split())
    S = data[1].strip()
    
    result = count_non_palindromic_permutations(N, K, S)
    print(result)

if __name__ == "__main__":
    main()