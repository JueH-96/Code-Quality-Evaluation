def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    s = data[1]
    # Array to store the maximum run length for each character
    max_run = [0] * 26
    run = 1
    # Traverse the string to compute runs
    for i in range(1, n):
        if s[i] == s[i-1]:
            run += 1
        else:
            idx = ord(s[i-1]) - ord('a')
            if run > max_run[idx]:
                max_run[idx] = run
            run = 1
    # Update for the last run
    idx = ord(s[-1]) - ord('a')
    if run > max_run[idx]:
        max_run[idx] = run
    # The answer is the sum of the maximum run lengths
    print(sum(max_run))

if __name__ == "__main__":
    main()