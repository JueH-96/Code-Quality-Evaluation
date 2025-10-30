def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    s = list(map(int, input[1:]))
    s.sort()
    count = 0
    for j in range(n):
        left = j - 1
        right = j + 1
        target = 2 * s[j]
        while left >= 0 and right < n:
            current_sum = s[left] + s[right]
            if current_sum == target:
                count += 1
                left -= 1
                right += 1
            elif current_sum < target:
                right += 1
            else:
                left -= 1
    print(count)

if __name__ == "__main__":
    main()