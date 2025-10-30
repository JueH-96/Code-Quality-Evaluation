import sys

def main():
    arr = []
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        num = int(line)
        arr.append(num)
        if num == 0:
            break

    for num in reversed(arr):
        print(num)

if __name__ == "__main__":
    main()