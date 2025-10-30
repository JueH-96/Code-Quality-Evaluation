import collections

def is_palindrome(s):
    n = len(s)
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            return False
    return True

def contains_palindrome_of_length_k(text, k):
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
    char_counts = collections.Counter(s)
    valid_permutation_count = 0
    
    def generate_permutations(current_permutation):
        nonlocal valid_permutation_count
        if len(current_permutation) == n:
            permutation_str = "".join(current_permutation)
            if not contains_palindrome_of_length_k(permutation_str, k):
                valid_permutation_count += 1
            return
            
        for char in sorted(char_counts.keys()):
            if char_counts[char] > 0:
                current_permutation.append(char)
                char_counts[char] -= 1
                generate_permutations(current_permutation)
                char_counts[char] += 1
                current_permutation.pop()
                
    generate_permutations([])
    print(valid_permutation_count)

if __name__ == '__main__':
    solve()