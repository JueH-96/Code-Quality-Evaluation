import sys

def main():
    input = sys.stdin.read
    N, M, *H = map(int, input().split())
    count = 0
    for h in H:
        if M >= h:
            M -= h
            count += 1
        else:
            break
    print(count)

if __name__ == "__main__":
    main()