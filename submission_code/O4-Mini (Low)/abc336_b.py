import sys

def main():
    data = sys.stdin.read().strip()
    if not data:
        return
    n = int(data)
    count = 0
    # Count how many times n is divisible by 2
    while n % 2 == 0:
        count += 1
        n //= 2
    print(count)

if __name__ == "__main__":
    main()