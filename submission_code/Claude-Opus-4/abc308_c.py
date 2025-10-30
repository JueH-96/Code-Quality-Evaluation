# YOUR CODE HERE
n = int(input())
people = []

for i in range(n):
    a, b = map(int, input().split())
    # Store (person_number, heads, total_tosses)
    people.append((i + 1, a, a + b))

# Sort by success rate (descending) and then by person number (ascending)
# To compare a1/(a1+b1) with a2/(a2+b2), we compare a1*(a2+b2) with a2*(a1+b1)
# For descending order, we want larger success rate first
def compare_key(person):
    person_num, heads, total = person
    # Return tuple for sorting: (-heads/total, person_num)
    # But to avoid division, we'll use a custom comparator
    return person

people.sort(key=lambda p: (-p[1] * 1.0 / p[2] if p[2] > 0 else 0, p[0]))

# Actually, let's use a more precise approach to avoid floating point issues
from functools import cmp_to_key

def compare(p1, p2):
    # p1 = (person_num1, heads1, total1)
    # p2 = (person_num2, heads2, total2)
    
    # Compare success rates: heads1/total1 vs heads2/total2
    # This is equivalent to: heads1 * total2 vs heads2 * total1
    cross1 = p1[1] * p2[2]
    cross2 = p2[1] * p1[2]
    
    if cross1 > cross2:
        return -1  # p1 has higher success rate, comes first
    elif cross1 < cross2:
        return 1   # p2 has higher success rate, comes first
    else:
        # Same success rate, sort by person number (ascending)
        return p1[0] - p2[0]

people.sort(key=cmp_to_key(compare))

# Print the sorted person numbers
print(' '.join(str(p[0]) for p in people))