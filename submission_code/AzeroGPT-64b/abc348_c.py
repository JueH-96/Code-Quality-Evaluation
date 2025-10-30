from collections import defaultdict
import bisect
n = int(input())
beans = []
for _ in range(n):
    a, c = map(int, input().split())
    beans.append((a, c))

# Sort by color, then by deliciousness
sorted_beans = sorted(beans, key=lambda x: (x[1], -x[0]))

# Group beans by color using defaultdict(list)
beans_by_color = defaultdict(list)
for a, c in sorted_beans:
    beans_by_color[c].append(a)

# Process colors to find possible options
options = []
for color, values in beans_by_color.items():
    if len(values) > 1:
        for _ in range(2):
            # Add the second deliciousness value for colors with more than one bean
            options.append(values[-2])
    elif options:
        # Special case for exactly one bean, use the last option
        options.append(options[-1])

# Find the maximum of possible options
if options:
    print(max(options))
else:
    # If no options, meaning all colors had only one bean, use the highest deliciousness of any bean
    print(max(beans, key=lambda x: x[0])[0])