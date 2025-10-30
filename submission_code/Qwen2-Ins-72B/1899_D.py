from collections import Counter
import math

def count_pairs(a):
    counter = Counter(a)
    result = 0
    for x in counter:
        for y in counter:
            if x == y:
                if x == 1 or x == 0:
                    continue
                if counter[x] > 1:
                    result += counter[x] * (counter[x] - 1) // 2
            else:
                if x * math.log2(y) == y * math.log2(x):
                    result += counter[x] * counter[y]
    return result

def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        print(count_pairs(a))

if __name__ == "__main__":
    main()