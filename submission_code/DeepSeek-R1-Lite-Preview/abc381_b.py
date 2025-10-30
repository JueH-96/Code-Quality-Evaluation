from collections import Counter
import sys

def main():
    S = sys.stdin.read().strip()
    
    # Check if the length of S is even
    if len(S) % 2 != 0:
        print("No")
        return
    
    # Check if each pair of consecutive characters is identical
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return
    
    # Count the frequency of each character
    count = Counter(S)
    
    # Check if each character appears exactly twice
    for char, freq in count.items():
        if freq != 2:
            print("No")
            return
    
    # If all checks pass, print "Yes"
    print("Yes")

if __name__ == "__main__":
    main()