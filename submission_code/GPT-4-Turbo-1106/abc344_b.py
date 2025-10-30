import sys

def main():
    numbers = []
    for line in sys.stdin:
        number = int(line.strip())
        numbers.append(number)
        if number == 0:
            break
    for number in reversed(numbers):
        print(number)

if __name__ == "__main__":
    main()