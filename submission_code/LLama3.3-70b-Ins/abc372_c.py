def count_substring(string, sub_string):
    """Count the occurrences of a substring in a string."""
    count = start = 0
    while True:
        start = string.find(sub_string, start)
        if start == -1:
            return count
        count += 1
        start += 1

def main():
    """Read input from stdin and solve the problem."""
    n, q = map(int, input().split())
    s = list(input())
    for _ in range(q):
        x, c = input().split()
        x = int(x) - 1  # Convert to 0-based index
        s[x] = c
        print(count_substring(''.join(s), 'ABC'))

if __name__ == "__main__":
    main()