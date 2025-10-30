import sys

def main() -> None:
    s = sys.stdin.readline().strip()
    vowels = set('aeiou')
    result = ''.join(ch for ch in s if ch not in vowels)
    print(result)

if __name__ == "__main__":
    main()