import sys

def main():
    q = int(sys.stdin.readline())
    a = []
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            a.append(x)
        else:
            k = int(parts[1])
            print(a[-k])

if __name__ == "__main__":
    main()