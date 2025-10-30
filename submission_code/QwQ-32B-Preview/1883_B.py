def can_form_palindrome(t, test_cases):
    results = []
    for _ in range(t):
        n, k = test_cases[_][0]
        s = test_cases[_][1]
        freq = [0] * 26
        for char in s:
            freq[ord(char) - ord('a')] += 1
        initial_odd = sum(1 for f in freq if f % 2 == 1)
        required_odd = (n - k) % 2
        if abs(initial_odd - required_odd) <= k and (initial_odd - required_odd) % 2 == k % 2:
            results.append("YES")
        else:
            results.append("NO")
    return results

# Example Usage
t = 14
test_cases = [
    [(1, 0), "a"],
    [(2, 0), "ab"],
    [(2, 1), "ba"],
    [(3, 1), "abb"],
    [(3, 2), "abc"],
    [(6, 2), "bacacd"],
    [(6, 2), "fagbza"],
    [(6, 2), "zwaafa"],
    [(7, 2), "taagaak"],
    [(14, 3), "ttrraakkttoorr"],
    [(5, 3), "debdb"],
    [(5, 4), "ecadc"],
    [(5, 3), "debca"],
    [(5, 3), "abaac"]
]
results = can_form_palindrome(t, test_cases)
for res in results:
    print(res)