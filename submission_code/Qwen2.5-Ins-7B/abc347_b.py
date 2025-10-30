# YOUR CODE HERE
def count_substrings(s):
    n = len(s)
    return n * (n + 1) // 2 - s.count('')
    
s = input().strip()
print(count_substrings(s))