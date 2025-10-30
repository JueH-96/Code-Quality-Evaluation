import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1

    scores = [0] * (N + 1)  # 1-based indexing
    freq = defaultdict(int)
    freq[0] = N
    ans = []

    for _ in range(T):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1

        old = scores[a]
        # Decrease the count of the old score
        freq[old] -= 1
        if freq[old] == 0:
            del freq[old]
        # Update the player's score
        new = old + b
        scores[a] = new
        # Increase the count of the new score
        freq[new] += 1
        # Record the current number of distinct scores
        ans.append(str(len(freq)))

    print('
'.join(ans))

if __name__ == "__main__":
    main()