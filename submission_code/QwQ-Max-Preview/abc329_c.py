import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    
    if n == 0:
        print(0)
        return
    
    max_counts = defaultdict(int)
    current_char = s[0]
    current_length = 1
    
    for c in s[1:]:
        if c == current_char:
            current_length += 1
        else:
            if current_length > max_counts[current_char]:
                max_counts[current_char] = current_length
            current_char = c
            current_length = 1
    
    # Update for the last character run
    if current_length > max_counts[current_char]:
        max_counts[current_char] = current_length
    
    print(sum(max_counts.values()))

if __name__ == "__main__":
    main()