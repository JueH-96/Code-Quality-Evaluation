# YOUR CODE HERE
def is_1122_string(S):
    # Condition 1: Length is even
    if len(S) % 2 != 0:
        return "No"
    
    # Condition 2: Each consecutive pair has equal characters
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            return "No"
    
    # Condition 3: Each character appears exactly twice
    from collections import Counter
    counts = Counter(S)
    for count in counts.values():
        if count != 2:
            return "No"
    
    return "Yes"

if __name__ == "__main__":
    import sys
    S = sys.stdin.read().strip()
    print(is_1122_string(S))