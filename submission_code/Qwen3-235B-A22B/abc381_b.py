import sys
from collections import Counter

def main():
    s = sys.stdin.readline().strip()
    
    # Check if length is even
    if len(s) % 2 != 0:
        print("No")
        return
    
    # Check each consecutive pair
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            print("No")
            return
    
    # Check each character appears exactly twice
    count = Counter(s)
    for char, freq in count.items():
        if freq != 2:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()