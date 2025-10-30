import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    if n == 0:
        print(0)
        return

    max_run = {}
    current_char = s[0]
    current_length = 1

    for i in range(1, len(s)):
        c = s[i]
        if c == current_char:
            current_length += 1
        else:
            # Update the max for the current run
            max_run[current_char] = max(max_run.get(current_char, 0), current_length)
            current_char = c
            current_length = 1

    # Process the last run
    max_run[current_char] = max(max_run.get(current_char, 0), current_length)

    total = sum(max_run.values())
    print(total)

if __name__ == "__main__":
    main()