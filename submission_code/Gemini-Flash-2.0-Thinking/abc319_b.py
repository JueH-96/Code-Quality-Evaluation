def solve():
    n = int(input())
    result = ""
    for i in range(n + 1):
        found_divisor = None
        for j in range(1, 10):
            if n % j == 0:
                if n // j == 0:
                    continue
                if i % (n // j) == 0:
                    if found_divisor is None or j < found_divisor:
                        found_divisor = j
        if found_divisor is not None:
            result += str(found_divisor)
        else:
            result += "-"
    print(result)

# YOUR CODE HERE
solve()