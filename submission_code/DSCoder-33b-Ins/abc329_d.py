# YOUR CODE HERE
import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    votes = list(map(int, sys.stdin.readline().split()))

    counts = [0] * (N + 1)
    max_count = 0
    max_candidate = 0

    for vote in votes:
        counts[vote] += 1
        if counts[vote] > max_count or (counts[vote] == max_count and vote < max_candidate):
            max_count = counts[vote]
            max_candidate = vote
        print(max_candidate)

if __name__ == "__main__":
    main()