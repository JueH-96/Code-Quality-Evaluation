# YOUR CODE HERE
def count_repetitive_substrings(s):
    count = 0
    n = len(s)
    
    for i in range(n):
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        length = j - i
        count += length
    
    return count

n = int(input())
s = input().strip()

result = count_repetitive_substrings(s)
print(result)