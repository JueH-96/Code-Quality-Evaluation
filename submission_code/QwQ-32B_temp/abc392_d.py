import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    dice = []
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        k_i = parts[0]
        a_list = parts[1:]
        count = Counter(a_list)
        dice.append((k_i, count))
    
    max_prob = 0.0
    for i in range(n):
        k1, cnt1 = dice[i]
        for j in range(i + 1, n):
            k2, cnt2 = dice[j]
            total = 0.0
            if len(cnt1) < len(cnt2):
                smaller = cnt1
                bigger = cnt2
            else:
                smaller = cnt2
                bigger = cnt1
            for x in smaller:
                if x in bigger:
                    total += (smaller[x] * bigger[x]) / (k1 * k2)
            if total > max_prob:
                max_prob = total
    
    print("{0:.12f}".format(max_prob))

if __name__ == "__main__":
    main()