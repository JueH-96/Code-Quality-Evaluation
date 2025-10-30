def can_make_abc(s):
    if s == "abc":
        return "YES"
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            swapped = list(s)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            if ''.join(swapped) == "abc":
                return "YES"
    return "NO"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    t = int(data[0])
    results = []

    for i in range(t):
        s = data[i + 1]
        results.append(can_make_abc(s))

    for result in results:
        print(result)

if __name__ == "__main__":
    main()