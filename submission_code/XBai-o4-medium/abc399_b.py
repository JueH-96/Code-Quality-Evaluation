from collections import Counter
import sys

def main():
    n = int(sys.stdin.readline())
    p = list(map(int, sys.stdin.readline().split()))
    count = Counter()
    for num in p:
        count[num] += 1
    sorted_scores = sorted(count.keys(), reverse=True)
    score_rank = {}
    r = 1
    for score in sorted_scores:
        score_rank[score] = r
        r += count[score]
    for num in p:
        print(score_rank[num])

if __name__ == "__main__":
    main()