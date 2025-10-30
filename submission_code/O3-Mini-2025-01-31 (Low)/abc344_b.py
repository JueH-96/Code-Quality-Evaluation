def main():
    import sys
    nums = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            num = int(line)
            nums.append(num)
            if num == 0:  # when read 0, stop reading further numbers
                break
    # We need to output in reverse order
    for num in reversed(nums):
        print(num)

if __name__ == '__main__':
    main()