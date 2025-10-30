import sys
from collections import Counter

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        k = int(input[idx+1])
        s = input[idx+2]
        idx +=3
        m = n - k
        counts = Counter(s)
        total_max_even = 0
        max_evens = []
        max_odds = []
        for c in counts.values():
            me = c if (c % 2 == 0) else (c - 1)
            mo = c if (c % 2 == 1) else (c - 1)
            max_evens.append(me)
            max_odds.append(mo)
            total_max_even += me
        if m % 2 == 0:
            if total_max_even >= m:
                print("YES")
            else:
                print("NO")
        else:
            found = False
            for i in range(len(max_evens)):
                me = max_evens[i]
                mo = max_odds[i]
                total_max_x = (total_max_even - me) + mo
                if total_max_x >= m:
                    found = True
                    break
            print("YES" if found else "NO")

if __name__ == "__main__":
    main()