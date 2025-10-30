# YOUR CODE HERE
def can_form_palindrome_after_removal(n, k, s):
    from collections import Counter
    
    # Count frequency of each character
    freq = Counter(s)
    
    # Count how many characters have an odd frequency
    odd_count = sum(1 for count in freq.values() if count % 2 != 0)
    
    # Calculate the length of the string after removing k characters
    remaining_length = n - k
    
    # Check if the remaining characters can form a palindrome
    if odd_count <= remaining_length:
        return "YES"
    else:
        return "NO"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    
    results = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        s = data[index + 2]
        index += 3
        
        result = can_form_palindrome_after_removal(n, k, s)
        results.append(result)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()