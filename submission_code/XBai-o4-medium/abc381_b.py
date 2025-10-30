import sys
from collections import Counter

def is_1122_string(s):
    if len(s) % 2 != 0:
        return "No"
    # Check each pair of characters
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            return "No"
    # Check each character appears exactly twice
    cnt = Counter(s)
    for v in cnt.values():
        if v != 2:
            return "No"
    return "Yes"

def main():
    s = sys.stdin.readline().strip()
    print(is_1122_string(s))

if __name__ == "__main__":
    main()