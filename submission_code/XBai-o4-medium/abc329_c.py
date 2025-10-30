import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    
    if n == 0:
        print(0)
        return
    
    max_run = defaultdict(int)
    current_char = s[0]
    current_length = 1
    
    for i in range(1, n):
        if s[i] == current_char:
            current_length += 1
        else:
            if current_length > max_run[current_char]:
                max_run[current_char] = current_length
            current_char = s[i]
            current_length = 1
    
    # Process the last run
    if current_length > max_run[current_char]:
        max_run[current_char] = current_length
    
    total = sum(max_run.values())
    print(total)

if __name__ == "__main__":
    main()