import sys
from collections import defaultdict

def main():
    N, Q = map(int, sys.stdin.readline().split())
    C = list(map(int, sys.stdin.readline().split()))
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    
    boxes = defaultdict(set)
    for i, c in enumerate(C):
        boxes[i+1].add(c)
    
    for a, b in queries:
        boxes[b] = boxes[b].union(boxes[a])
        boxes[a] = set()
        print(len(boxes[b]))

if __name__ == "__main__":
    main()