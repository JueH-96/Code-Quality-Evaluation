def main():
    import sys
    data = sys.stdin.read().strip().split()
    # There should be exactly 8 integers according to the problem specification
    if len(data) != 8:
        print("No")
        return
    
    try:
        s = list(map(int, data))
    except ValueError:
        print("No")
        return

    # Condition 1: Monotonically non-decreasing order (S_1 <= S_2 <= ... <= S_8)
    for i in range(1, len(s)):
        if s[i] < s[i - 1]:
            print("No")
            return

    # Condition 2 and 3: Each number should be between 100 and 675 and a multiple of 25.
    for number in s:
        if not (100 <= number <= 675) or (number % 25 != 0):
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()