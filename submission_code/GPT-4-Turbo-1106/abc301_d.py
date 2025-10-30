def generate_combinations(s, index, current_value, n, results):
    if index == len(s):
        if current_value <= n:
            results.append(current_value)
        return

    if s[index] == '?':
        generate_combinations(s, index + 1, current_value * 2, n, results)
        generate_combinations(s, index + 1, current_value * 2 + 1, n, results)
    else:
        digit = int(s[index])
        generate_combinations(s, index + 1, current_value * 2 + digit, n, results)

def find_max_value_less_than_n(s, n):
    results = []
    generate_combinations(s, 0, 0, n, results)
    return max(results) if results else -1

# Read input
s = input().strip()
n = int(input().strip())

# Find and print the answer
print(find_max_value_less_than_n(s, n))