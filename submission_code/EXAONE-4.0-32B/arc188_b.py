import math
import sys

def is_power_of_two(x):
    if x == 0:
        return False
    while x % 2 == 0:
        x //= 2
    return x == 1

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        index += 2
        g = math.gcd(n, k)
        if is_power_of_two(g):
            results.append("Yes")
        else:
            results.append("No")
    print("
".join(results))

if __name__ == "__main__":
    main()