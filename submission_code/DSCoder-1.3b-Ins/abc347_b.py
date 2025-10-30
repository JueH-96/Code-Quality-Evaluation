import sys

def count_substrings(s):
    n = len(s)
    count = 0
    for i in range(n):
        for j in range(i+1, n+1):
            count += 1
    return count

def main():
    s = sys.stdin.readline().strip()
    print(count_substrings(s))

if __name__ == "__main__":
    main()