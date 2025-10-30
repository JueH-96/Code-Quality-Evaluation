import sys
input = sys.stdin.read

def main():
    N = int(input().strip())
    dragon_string = 'L' + 'o' * N + 'ng'
    print(dragon_string)

if __name__ == "__main__":
    main()