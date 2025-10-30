# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    T = int(data[idx+1])
    idx += 2
    A = []
    B = []
    for _ in range(T):
        A.append(int(data[idx]))
        B.append(int(data[idx+1]))
        idx += 2
    
    # Initialize scores
    scores = [0] * (N + 1)  # 1-based indexing
    freq = defaultdict(int)
    freq[0] = N
    unique = 1  # Initially, only 0 is present
    
    for i in range(T):
        a = A[i]
        b = B[i]
        old_score = scores[a]
        new_score = old_score + b
        scores[a] = new_score
        
        # Update frequency of old_score
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
        
        # Update frequency of new_score
        if new_score in freq:
            freq[new_score] += 1
        else:
            freq[new_score] = 1
        
        # The number of unique scores is the number of keys in freq
        print(len(freq))

if __name__ == "__main__":
    main()