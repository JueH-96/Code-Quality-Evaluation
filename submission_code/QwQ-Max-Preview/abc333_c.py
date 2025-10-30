n = int(input())
max_length = 50  # Adjusted to ensure coverage for large N
sums = set()

for a in range(1, max_length + 1):
    for b in range(a, max_length + 1):
        for c in range(b, max_length + 1):
            max_d = max(a, b, c) - 1
            digits = []
            for d in range(max_d, -1, -1):
                count = 0
                if a > d:
                    count += 1
                if b > d:
                    count += 1
                if c > d:
                    count += 1
                digits.append(str(count))
            sum_str = ''.join(digits)
            sum_val = int(sum_str)
            sums.add(sum_val)

sorted_sums = sorted(sums)
print(sorted_sums[n-1])