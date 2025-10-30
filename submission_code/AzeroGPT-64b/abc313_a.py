# We import the numpy library as np for mathematical operations if needed.
import numpy as np

# We read an integer N from the standard input and convert it to an integer using int().
n = int(input())

# We read a space-separated list of integers P of length N, convert each element to an integer using int(),
# store the list in a numpy array, and then access its elements as needed.
p = np.array(input().split(), dtype=int)

# The given condition requires finding the minimum non-negative integer x so that P[0] + x > P[i] for all i != 0.
# We implement this logic as follows:
# 1. We calculate the maximum programming ability of people excluding person 1, np.max(p[1:]),
#    and adjust it to be one point lower than P[0] + x, i.e., max_p_excl_1 = np.max(p[1:]) + 1.
# 2. We return the minimum x such that P[0] + x equals max_p_excl_1, i.e., x = max_p_excl_1 - P[0],
#    ensuring that the result is non-negative.
ans = max(0, np.max(p[1:]) + 1 - p[0])

# We output the answer as an integer to the standard output.
print(ans)