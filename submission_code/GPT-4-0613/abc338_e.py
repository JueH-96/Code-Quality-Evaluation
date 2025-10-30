import sys
from collections import deque
from operator import itemgetter

def main():
    N = int(input())
    chords = [list(map(int, input().split())) for _ in range(N)]
    chords = sorted([(min(a, b), max(a, b)) for a, b in chords], key=itemgetter(1))

    stack = deque()
    for a, b in chords:
        while stack and stack[-1] < a:
            stack.pop()
        if stack and stack[-1] > a:
            print('Yes')
            return
        stack.append(b)
    print('No')

if __name__ == '__main__':
    main()