# YOUR CODE HERE
from math import factorial
from collections import Counter

def is_palindrome(s, k):
    n = len(s)
    for i in range(n - k + 1):
        if s[i:i + k] == s[i:i + k][::-1]:
            return True
    return False

def count_valid_permutations(s, k):
    n = len(s)
    count = Counter(s)
    total_permutations = factorial(n)
    for char, freq in count.items():
        total_permutations //= factorial(freq)
    
    invalid_permutations = 0
    for i in range(n):
        for j in range(i + 1, n):
            temp_s = list(s)
            temp_s[i], temp_s[j] = temp_s[j], temp_s[i]
            if is_palindrome(''.join(temp_s), k):
                invalid_permutations += 1
                break
    
    return total_permutations - invalid_permutations

if __name__ == "__main__":
    n, k = map(int, input().split())
    s = input().strip()
    print(count_valid_permutations(s, k))