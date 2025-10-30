from collections import Counter

def is_palindrome(s):
    n = len(s)
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            return False
    return True

def contains_palindrome(text, k):
    n = len(text)
    if k > n:
        return False
    for i in range(n - k + 1):
        substring = text[i:i+k]
        if is_palindrome(substring):
            return True
    return False

def solve():
    n, k = map(int, input().split())
    s = input()
    initial_char_counts = Counter(s)
    valid_permutation_count = 0
    
    def get_permutations(current_permutation_list, remaining_length, char_counts):
        nonlocal valid_permutation_count
        if remaining_length == 0:
            permutation_str = "".join(current_permutation_list)
            if not contains_palindrome(permutation_str, k):
                valid_permutation_count += 1
            return
            
        for char_code in range(ord('a'), ord('z') + 1):
            char = chr(char_code)
            if char_counts[char] > 0:
                current_permutation_list.append(char)
                char_counts[char] -= 1
                get_permutations(current_permutation_list, remaining_length - 1, char_counts)
                char_counts[char] += 1
                current_permutation_list.pop()
                
    get_permutations([], n, initial_char_counts.copy())
    print(valid_permutation_count)

if __name__ == '__main__':
    solve()