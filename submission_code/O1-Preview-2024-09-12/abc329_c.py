# YOUR CODE HERE
import sys

def main():
    import sys
    input_lines = sys.stdin.read().split()
    N = int(input_lines[0])
    S = input_lines[1]

    MaxLen = {}
    current_char = None
    current_len = 0

    for c in S:
        if c == current_char:
            current_len += 1
        else:
            if current_char is not None:
                MaxLen[current_char] = max(MaxLen.get(current_char, 0), current_len)
            current_char = c
            current_len = 1
    # Update for the last character
    MaxLen[current_char] = max(MaxLen.get(current_char, 0), current_len)

    total_substrings = sum(MaxLen.values())

    print(total_substrings)

if __name__ == "__main__":
    main()