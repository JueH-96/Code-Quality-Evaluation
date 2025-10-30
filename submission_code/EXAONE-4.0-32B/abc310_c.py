n = int(input().strip())
distinct_set = set()
for _ in range(n):
    s = input().strip()
    rev = s[::-1]
    canonical = s if s <= rev else rev
    distinct_set.add(canonical)
print(len(distinct_set))