def solve():
    w, b = map(int, input().split())

    base_string = "wbwbwwbwbwbw"
    long_string = base_string * 300  # Generate a sufficiently long string

    n = len(long_string)
    for i in range(n):
        for j in range(i, n):
            substring = long_string[i:j+1]
            w_count = substring.count('w')
            b_count = substring.count('b')
            if w_count == w and b_count == b:
                print("Yes")
                return

    print("No")

solve()