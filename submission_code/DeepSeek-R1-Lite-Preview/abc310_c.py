import sys

def main():
    input = sys.stdin.read().splitlines()
    N = int(input[0])
    normalized_set = set()
    for i in range(1, N + 1):
        s = input[i]
        s_rev = s[::-1]
        normalized = min(s, s_rev)
        normalized_set.add(normalized)
    print(len(normalized_set))

if __name__ == "__main__":
    main()