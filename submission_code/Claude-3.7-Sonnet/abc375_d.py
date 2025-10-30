# YOUR CODE HERE
def count_palindrome_triples(S):
    count = 0
    n = len(S)
    
    for i in range(n):
        for k in range(i+2, n):  # k must be at least i+2 to ensure j exists between i and k
            if S[i] == S[k]:
                # For each matching pair of characters, count how many positions j can take
                # There are (k-i-1) positions between i and k
                count += (k - i - 1)
    
    return count

S = input().strip()
print(count_palindrome_triples(S))