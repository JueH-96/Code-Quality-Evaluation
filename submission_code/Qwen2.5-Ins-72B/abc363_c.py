from itertools import permutations

def is_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_valid_permutations(n, k, s):
    unique_permutations = set(permutations(s))
    count = 0
    for perm in unique_permutations:
        perm_str = ''.join(perm)
        if not is_palindrome(perm_str, k):
            count += 1
    return count

if __name__ == "__main__":
    n, k = map(int, input().split())
    s = input().strip()
    print(count_valid_permutations(n, k, s))