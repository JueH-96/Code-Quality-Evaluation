# YOUR CODE HERE

import sys

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    photos = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    return N, M, photos

def count_bad_mood_pairs(N, M, photos):
    bad_mood_pairs = 0
    for i in range(M-1):
        for j in range(N):
            for k in range(j+1, N):
                if photos[i][j] == photos[i+1][k] or photos[i][k] == photos[i+1][j]:
                    bad_mood_pairs += 1
    return bad_mood_pairs

def main():
    N, M, photos = read_input()
    bad_mood_pairs = count_bad_mood_pairs(N, M, photos)
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()