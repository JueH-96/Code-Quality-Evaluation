import sys
from collections import Counter

def main():
    input = sys.stdin.read
    data = list(map(int, input().split()))
    
    # Count the occurrences of each number
    counts = Counter(data)
    
    # We need exactly one triplet and one pair
    found_triplet = False
    found_pair = False
    
    for count in counts.values():
        if count >= 3:
            if not found_triplet:
                found_triplet = True
            elif count >= 5:
                # If there's a count of 5 or more, it can also serve as a pair after serving as a triplet
                found_pair = True
        if count >= 2:
            if not found_pair:
                found_pair = True
    
    # If we have found both a triplet and a pair
    if found_triplet and found_pair:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()