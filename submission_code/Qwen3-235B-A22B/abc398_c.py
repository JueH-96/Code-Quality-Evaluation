import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    cnt = Counter(a)
    max_val = -float('inf')
    label = -1
    for i in range(n):
        val = a[i]
        if cnt[val] == 1:
            if val > max_val:
                max_val = val
                label = i + 1  # Convert to 1-based index
    print(label if label != -1 else -1)

if __name__ == "__main__":
    main()