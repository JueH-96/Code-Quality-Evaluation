def can_form_palindrome(t, test_cases):
    results = []
    for case in test_cases:
        n, k, s = case
        m = n - k
        if m <= 0:
            results.append("NO")
            continue
        counts = [0] * 26
        for c in s:
            counts[ord(c) - 97] += 1
        O = sum(1 for cnt in counts if cnt % 2 != 0)
        required_odds = m % 2
        delta = required_odds - O
        if (delta % 2 != k % 2) or abs(delta) > k:
            results.append("NO")
        else:
            results.append("YES")
    return results

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    test_cases = []
    for _ in range(t):
        n = int(input[ptr])
        k = int(input[ptr+1])
        ptr +=2
        s = input[ptr]
        ptr +=1
        test_cases.append( (n, k, s) )
    results = can_form_palindrome(t, test_cases)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()