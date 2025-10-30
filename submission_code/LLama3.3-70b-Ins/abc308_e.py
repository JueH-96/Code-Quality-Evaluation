def mex(a, b, c):
    """Calculate the minimum non-negative integer that equals neither a, b, nor c."""
    for i in range(3):
        if i not in (a, b, c):
            return i

def main():
    """Read the inputs from stdin, solve the problem, and write the answer to stdout."""
    n = int(input())
    a = list(map(int, input().split()))
    s = input()

    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[i] + s[j] + s[k] == "MEX":
                    total += mex(a[i], a[j], a[k])

    print(total)

if __name__ == "__main__":
    main()