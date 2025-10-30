import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    wins = []
    for i in range(1, N+1):
        s = input().strip()
        w = s.count('o')
        wins.append((w, i))
    # Sort by wins descending, then player number ascending
    wins.sort(key=lambda x: (-x[0], x[1]))
    result = [str(player) for _, player in wins]
    print(" ".join(result))

if __name__ == "__main__":
    main()