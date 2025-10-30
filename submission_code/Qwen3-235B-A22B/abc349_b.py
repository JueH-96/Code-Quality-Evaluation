import sys
from collections import Counter

def main():
    s = sys.stdin.readline().strip()
    char_counts = Counter(s)
    freq_of_counts = Counter(char_counts.values())
    
    for count in freq_of_counts.values():
        if count != 2:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()