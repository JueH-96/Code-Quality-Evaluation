n, k = map(int, input().split())
funcs = [tuple(map(int, input().split())) for _ in range(n)]

# Custom comparator to sort functions in descending order
# such that f comes before g if f.B * g.A > g.B * f.A
def compare(f, g):
    return f[1] * g[0] > g[1] * f[0]

# Sort the functions in descending order based on the comparator
funcs.sort(key=lambda x: (-x[0], -x[1]))  # Initial sort by A and B to handle ties, but the compare function is used

# Custom sorting with the comparator (Python's sort is stable, so we need to sort with a key that reflects the comparator)
# To handle this, we use a lambda that returns a tuple for sorting
# The comparator f < g is equivalent to f.B * g.A > g.B * f.A
# We can sort by the key (f.B * g.A - g.B * f.A) in descending order, but since we can't compute pairwise, we use a custom approach
# Instead, we can use the ratio -f.B / f.A (to maximize B/A ratio)
# However, to avoid division and floating points, we can sort using the cross product
# So, we'll sort the functions in descending order of (B_i, A_i) using the cross product as the key
funcs.sort(key=lambda x: (-x[1]/x[0] if x[0] !=0 else 0), reverse=False)
# Wait, the correct way is to sort using the comparator. However, Python's sort doesn't support a custom comparator directly for large N.
# To handle this, we can use a lambda that returns a tuple for each function that can be used to sort them according to the comparator.
# The comparator f should come before g if f.B * g.A > g.B * f.A.
# To sort in descending order, we can use a key that is the negative of this value.
# However, since we can't compute pairwise, we need to find a key that reflects this.
# The correct way is to use a key that for each function f, returns (-f.B, f.A), which sorts in the desired order.
# Because, for two functions f and g:
# f.B * g.A > g.B * f.A  =>  f.B/f.A > g.B/g.A (if A's are positive)
# So sorting by B/A in descending order would achieve this.
# Since all A_i are >=1, we can compute B_i / A_i and sort in descending order.
funcs.sort(key=lambda x: (-x[1]/x[0]))

selected = funcs[:k]

# Apply the functions in the sorted order (selected is already sorted)
x = 1
for a, b in selected:
    x = a * x + b

print(x)