import sys

def can_form_abc(s):
    if s == "abc":
        return "YES"
    if s == "acb" or s == "bac" or s == "cba":
        return "YES"
    return "NO"

def main():
    input = sys.stdin.read
    data = input().split()

    t = int(data[0])
    results = []

    for i in range(t):
        s = data[i + 1]
        results.append(can_form_abc(s))

    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()