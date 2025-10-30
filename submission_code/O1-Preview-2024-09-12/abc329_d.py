# YOUR CODE HERE
import sys
import threading

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    counts = [0] * (N + 1)
    max_votes = 0
    winner = N + 1  # Initialize to a value larger than any candidate number

    for a in A:
        counts[a] += 1
        if counts[a] > max_votes:
            max_votes = counts[a]
            winner = a
        elif counts[a] == max_votes:
            winner = min(winner, a)
        print(winner)

threading.Thread(target=main).start()