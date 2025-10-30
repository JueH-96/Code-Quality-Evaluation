import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    numbers = list(map(int, data))
    for number in reversed(numbers):
        print(number)

if __name__ == "__main__":
    main()