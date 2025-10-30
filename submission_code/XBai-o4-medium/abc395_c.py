from collections import defaultdict
import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    pos = defaultdict(list)
    for idx, num in enumerate(a):
        pos[num].append(idx)
    min_len = float('inf')
    for key in pos:
        lst = pos[key]
        for i in range(1, len(lst)):
            diff = lst[i] - lst[i-1]
            current_len = diff + 1
            if current_len < min_len:
                min_len = current_len
    if min_len == float('inf'):
        print(-1)
    else:
        print(min_len)

if __name__ == "__main__":
    main()