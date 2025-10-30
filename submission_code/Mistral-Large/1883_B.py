import sys
from collections import Counter

def can_form_palindrome(s, k):
    freq = Counter(s)
    odd_count = sum(1 for count in freq.values() if count % 2 != 0)
    if odd_count <= k + (k % 2):
        return "YES"
    else:
        return "NO"

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    t = int(data[index])
    index += 1
    results = []

    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        index += 2
        s = data[index]
        index += 1
        results.append(can_form_palindrome(s, k))

    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()