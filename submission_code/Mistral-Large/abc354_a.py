import sys

def find_day(H):
    height = 0
    day = 0
    while height <= H:
        day += 1
        height += 2 ** (day - 1)
    return day

if __name__ == "__main__":
    input = sys.stdin.read()
    H = int(input.strip())
    print(find_day(H))