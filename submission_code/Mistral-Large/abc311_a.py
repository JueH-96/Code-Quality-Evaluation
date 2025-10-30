import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    seen = set()
    for i, char in enumerate(S):
        seen.add(char)
        if 'A' in seen and 'B' in seen and 'C' in seen:
            print(i + 1)
            break

if __name__ == "__main__":
    main()