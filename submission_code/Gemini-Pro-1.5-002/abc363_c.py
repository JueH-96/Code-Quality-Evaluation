# YOUR CODE HERE
import math

def is_palindrome(s):
    return s == s[::-1]

def solve():
    n, k = map(int, input().split())
    s = input()
    
    chars = sorted(list(s))
    permutations = set()
    
    def generate_permutations(current_permutation):
        if len(current_permutation) == n:
            permutations.add("".join(current_permutation))
            return
        
        for i in range(len(chars)):
            if chars[i] is not None:
                temp = chars[i]
                chars[i] = None
                generate_permutations(current_permutation + [temp])
                chars[i] = temp

    generate_permutations([])
    
    count = 0
    for p in permutations:
        valid = True
        for i in range(n - k + 1):
            substring = p[i:i+k]
            if is_palindrome(substring):
                valid = False
                break
        if valid:
            count += 1
            
    print(count)

solve()