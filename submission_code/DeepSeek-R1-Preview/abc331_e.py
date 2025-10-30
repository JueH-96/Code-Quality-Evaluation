n, m, l = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Create a list of tuples (value, index) for a and sort them in descending order
a_with_indices = [(a[i], i + 1) for i in range(n)]
a_with_indices.sort(key=lambda x: (-x[0], x[1]))

# Similarly for b
b_with_indices = [(b[i], i + 1) for i in range(m)]
b_with_indices.sort(key=lambda x: (-x[0], x[1]))

# Take the top 200 elements from each list
K = 200
top_a = a_with_indices[:K]
top_b = b_with_indices[:K]

# Read forbidden pairs and store them in a set
forbidden = set()
for _ in range(l):
    c, d = map(int, input().split())
    forbidden.add((c, d))

max_sum = 0

# Check all possible pairs in the top K a and top K b
for a_val, a_idx in top_a:
    for b_val, b_idx in top_b:
        if (a_idx, b_idx) not in forbidden:
            current_sum = a_val + b_val
            if current_sum > max_sum:
                max_sum = current_sum

print(max_sum)