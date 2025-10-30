def main():
    numbers = []

    def dfs(current, digit_sum):
        if digit_sum >= 3:
            numbers.append(int(''.join(map(str, current))))
        if len(current) >= 13:  # Maximum length based on sample input 3
            return
        last = current[-1] if current else 0
        start = 1 if not current else last
        for d in range(start, 4):
            dfs(current + [d], digit_sum + d)

    dfs([], 0)
    numbers = sorted(numbers)
    
    n = int(input())
    print(numbers[n-1])

main()