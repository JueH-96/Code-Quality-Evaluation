import sys
from collections import Counter

def main():
    S = input().strip()
    counter = Counter(S)
    max_count = max(counter.values())
    most_common_chars = [char for char, count in counter.items() if count == max_count]
    most_common_chars.sort()
    print(most_common_chars[0])

if __name__ == "__main__":
    main()