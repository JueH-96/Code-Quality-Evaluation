# YOUR CODE HERE
def min_insertions_to_valid_sequence(S):
    n = len(S)
    # Determine the target length, which should be even
    target_length = n if n % 2 == 0 else n + 1
    
    # Count mismatches
    mismatches = 0
    for i in range(n):
        expected_char = 'i' if i % 2 == 0 else 'o'
        if S[i] != expected_char:
            mismatches += 1
    
    # Calculate the number of insertions needed
    # Insertions needed to fix mismatches + (if length is odd, we need one more insertion to make it even)
    insertions_needed = mismatches + (target_length - n)
    
    return insertions_needed

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    print(min_insertions_to_valid_sequence(S))