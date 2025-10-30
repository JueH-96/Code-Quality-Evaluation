import sys

# Read input from Standard Input
line = sys.stdin.readline().split()
A = int(line[0])
B = int(line[1])

# Determine the number of integers x that satisfy the condition.
# An arithmetic sequence of three integers p, q, r satisfies the property 2q = p + r.
# We are given two integers A and B, and we seek an integer x such that the set of three numbers {A, B, x}
# can be ordered to form an arithmetic sequence p, q, r.
# This means one of A, B, or x must be the middle term (q), and the other two must be the first and last terms (p and r).

# Case 1: x is the middle term (q = x).
# The other two terms must be A and B (in any order). So, p, r = A, B or p, r = B, A.
# The condition 2q = p + r becomes 2 * x = A + B.
# Thus, x = (A + B) / 2. This value of x is an integer if and only if (A + B) is even.

# Case 2: A is the middle term (q = A).
# The other two terms must be B and x. So, p, r = B, x or p, r = x, B.
# The condition 2q = p + r becomes 2 * A = B + x.
# Thus, x = 2 * A - B. This value of x is always an integer since A and B are integers.

# Case 3: B is the middle term (q = B).
# The other two terms must be A and x. So, p, r = A, x or p, r = x, A.
# The condition 2q = p + r becomes 2 * B = A + x.
# Thus, x = 2 * B - A. This value of x is always an integer since A and B are integers.

# The possible integer values for x are the distinct integers found in these three cases.
# Let v1 = (A + B) / 2, v2 = 2 * A - B, v3 = 2 * B - A.

# If A == B:
# The potential values for x are:
# v1 = (A + A) / 2 = A. Since 2A is always even, this is always an integer.
# v2 = 2 * A - A = A.
# v3 = 2 * A - A = A.
# All three potential values are A. The set of distinct integer values for x is {A}. The count is 1.

# If A != B:
# The values v2 = 2 * A - B and v3 = 2 * B - A are always integers.
# Are they distinct? If 2 * A - B = 2 * B - A, then 3 * A = 3 * B, which implies A = B.
# Since we are in the case A != B, v2 and v3 must be distinct integers.
# So, there are at least two distinct integer values for x (2*A - B and 2*B - A).

# Now, consider v1 = (A + B) / 2.
# If (A + B) is odd, v1 is not an integer. The only integer values for x are v2 and v3.
# Since A != B, v2 and v3 are distinct. The count of distinct integer values is 2.
# If (A + B) is even, v1 is an integer.
# Is v1 distinct from v2 and v3 when A != B?
# v1 == v2 => (A + B) / 2 = 2 * A - B => A + B = 4 * A - 2 * B => 3 * B = 3 * A => B = A. This contradicts A != B.
# v1 == v3 => (A + B) / 2 = 2 * B - A => A + B = 4 * B - 2 * A => 3 * A = 3 * B => A = B. This contradicts A != B.
# So, if A != B and (A + B) is even, v1 is a third distinct integer value, different from v2 and v3.
# The set of distinct integer values is { (A + B) / 2, 2 * A - B, 2 * B - A }. The count is 3.

# The condition "(A + B) is even" is equivalent to "A and B have the same parity".
# A and B have the same parity if A % 2 == B % 2.

if A == B:
    # Case 1: A = B. Only x = A is possible.
    count = 1
elif (A % 2) == (B % 2):
    # Case 2: A != B and A, B have the same parity. (A+B)/2 is integer and distinct from 2A-B and 2B-A.
    # Three distinct integer values: (A+B)/2, 2A-B, 2B-A.
    count = 3
else:
    # Case 3: A != B and A, B have different parity. (A+B)/2 is not integer.
    # Two distinct integer values: 2A-B, 2B-A.
    count = 2

# Print the result to Standard Output
print(count)