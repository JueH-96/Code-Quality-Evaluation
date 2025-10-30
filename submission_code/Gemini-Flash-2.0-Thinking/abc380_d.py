def solve():
    s = input()
    q = int(input())
    k_values = list(map(int, input().split()))

    n = len(s)

    def swap_case(char):
        if 'a' <= char <= 'z':
            return char.upper()
        elif 'A' <= char <= 'Z':
            return char.lower()
        return char

    results = []
    for k in k_values:
        k -= 1  # Convert to 0-based index
        rem = k % (2 * n)
        if rem < n:
            results.append(s[rem])
        else:
            results.append(swap_case(s[rem - n]))

    print(*results)

solve()