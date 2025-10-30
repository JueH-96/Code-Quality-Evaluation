# YOUR CODE HERE
def is_good_string(S):
    from collections import Counter
    
    # Count the frequency of each character in the string
    freq = Counter(S)
    
    # Count how many characters have each frequency
    freq_count = Counter(freq.values())
    
    # Check the condition for a good string
    for count in freq_count.values():
        if count != 0 and count != 2:
            return "No"
    
    return "Yes"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    print(is_good_string(S))