import sys

def main():
    n = int(sys.stdin.readline())
    strings = [sys.stdin.readline().strip() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                combined = strings[i] + strings[j]
                if combined == combined[::-1]:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()