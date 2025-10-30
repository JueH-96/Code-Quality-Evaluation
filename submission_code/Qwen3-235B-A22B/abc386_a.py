from collections import Counter
import sys

def main():
    A, B, C, D = map(int, sys.stdin.readline().split())
    counts = Counter([A, B, C, D])
    for x in range(1, 14):
        temp = counts.copy()
        temp[x] += 1
        vals = sorted(temp.values())
        if vals == [2, 3]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()