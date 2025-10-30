import sys
from collections import defaultdict

def main():
    n, t = map(int, sys.stdin.readline().split())
    scores = [0] * n
    freq = defaultdict(int)
    freq[0] = n
    for _ in range(t):
        a, b = map(int, sys.stdin.readline().split())
        index = a - 1
        s = scores[index]
        # Update the frequency of the old score
        freq[s] -= 1
        if freq[s] == 0:
            del freq[s]
        # Calculate new score and update
        new_s = s + b
        scores[index] = new_s
        freq[new_s] += 1
        # Output the number of distinct scores
        print(len(freq))

if __name__ == "__main__":
    main()