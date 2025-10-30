def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return

    N = int(data[0])                 # Number of integers (not strictly needed after reading)
    nums = list(map(int, data[1:1+N]))

    max_val = max(nums)              # The overall largest value
    second_max = max(x for x in nums if x != max_val)  # Largest value that isn't the overall max
    
    print(second_max)


if __name__ == "__main__":
    main()