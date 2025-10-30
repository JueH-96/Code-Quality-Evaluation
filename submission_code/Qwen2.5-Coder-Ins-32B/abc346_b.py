# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().strip()
    W, B = map(int, input.split())
    
    # The repeating pattern
    pattern = "wbwbwwbwbwbw"
    
    # Check all possible substrings of the pattern
    for i in range(len(pattern)):
        for j in range(i + 1, len(pattern) + 1):
            substring = pattern[i:j]
            if substring.count('w') == W and substring.count('b') == B:
                print("Yes")
                return
    
    # Check if the pattern can be extended to find the required substring
    extended_pattern = pattern * 2  # Extend the pattern to cover more cases
    for i in range(len(extended_pattern) - (W + B) + 1):
        substring = extended_pattern[i:i + W + B]
        if substring.count('w') == W and substring.count('b') == B:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()