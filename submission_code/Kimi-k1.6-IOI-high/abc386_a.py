import sys
from collections import Counter

def main():
    a, b, c, d = map(int, sys.stdin.readline().split())
    cards = [a, b, c, d]
    found = False
    for x in range(1, 14):
        temp = cards + [x]
        cnt = Counter(temp)
        values = sorted(cnt.values())
        if values == [2, 3]:
            found = True
            break
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()