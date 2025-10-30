import sys

def is_palindrome(t: str) -> bool:
    return t == t[::-1]

def main() -> None:
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    strings = data[1:]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_palindrome(strings[i] + strings[j]):
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()