# YOUR CODE HERE
def find_first_day(H):
    i = 0
    while True:
        if (2 ** i) - 1 > H:
            return i
        i += 1

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    H = int(input().strip())
    print(find_first_day(H))