from collections import Counter

def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Count frequency of each character
    freq = Counter(S)
    
    # Count how many characters have the same frequency
    freq_of_freq = Counter(freq.values())
    
    # Check the condition for good string
    for count in freq_of_freq.values():
        if count != 2 and count != 0:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()