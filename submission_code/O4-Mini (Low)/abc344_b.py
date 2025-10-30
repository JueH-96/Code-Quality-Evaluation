import sys

def main():
    nums = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        x = int(line)
        nums.append(x)
        if x == 0:
            break
    for x in reversed(nums):
        print(x)

if __name__ == "__main__":
    main()