import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    total = 0
    current_term = 1
    # Handle the last character (i = n in 1-based)
    digit = int(s[-1])
    total += digit * n * current_term
    # Iterate from n-1 down to 1 (1-based)
    for i in range(n-1, 0, -1):
        current_term = current_term * 10 + 1
        digit = int(s[i-1])
        total += digit * i * current_term
    print(total)

if __name__ == "__main__":
    main()