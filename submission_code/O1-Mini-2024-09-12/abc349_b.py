from collections import Counter
import sys

def main():
    S = sys.stdin.read().strip()
    freq = Counter(S)
    max_freq = max(freq.values())
    
    for i in range(1, max_freq + 1):
        count = sum(1 for v in freq.values() if v == i)
        if count != 0 and count != 2:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()