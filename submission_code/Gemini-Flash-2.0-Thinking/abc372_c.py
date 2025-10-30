def solve():
    n, q = map(int, input().split())
    s = list(input())

    def count_abc(text):
        count = 0
        for i in range(len(text) - 2):
            if text[i:i+3] == "ABC":
                count += 1
        return count

    for _ in range(q):
        x, c = input().split()
        x = int(x)
        s[x - 1] = c
        updated_s = "".join(s)
        print(count_abc(updated_s))

# Call the solve function
solve()