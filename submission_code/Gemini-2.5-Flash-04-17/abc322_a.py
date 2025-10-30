# Read N
N = int(input())

# Read S
S = input()

# Find the first occurrence of "ABC" in S.
# The find() method returns the lowest index in the string where the substring is found,
# or -1 if the substring is not found.
# This index is 0-based.
zero_indexed_position = S.find("ABC")

# The problem asks for the position (n) as a 1-indexed integer.
# If "ABC" is found starting at 0-indexed position 'i', the problem defines the position 'n'
# as the smallest integer satisfying 1 <= n <= N-2 and the substring from the n-th
# through (n+2)-th characters is "ABC".
# The n-th character is at 0-indexed position n-1.
# The (n+2)-th character is at 0-indexed position (n+2)-1 = n+1.
# The substring from n-th to (n+2)-th characters corresponds to S[n-1 : (n+1)+1] = S[n-1 : n+2].
# The condition is S[n-1 : n+2] == "ABC".
# If S.find("ABC") returns 'i', it means the first occurrence of "ABC" starts at 0-indexed 'i'.
# So, we have S[i : i+3] == "ABC".
# Comparing this with S[n-1 : n+2] == "ABC", we see that n-1 must be equal to i.
# Thus, the 1-indexed position n is i + 1.
# The find() method returns the *lowest* index, ensuring we find the *smallest* integer n.
# The constraints 1 <= n <= N-2 correspond to 0 <= n-1 <= N-3, which is 0 <= i <= N-3.
# A 3-character substring starting at index i needs indices i, i+1, i+2 to be valid.
# This requires i+2 < N, or i <= N-3. So, the 0-indexed position i must be <= N-3.
# The find() method will only return an index up to N-3 for a substring of length 3 in a string of length N.
# Thus, if find() returns i >= 0, then i is in the range [0, N-3], and i+1 is in the range [1, N-2],
# which satisfies the required range for n.

if zero_indexed_position != -1:
    # If "ABC" was found, convert the 0-indexed position to a 1-indexed position.
    one_indexed_position = zero_indexed_position + 1
    print(one_indexed_position)
else:
    # If "ABC" was not found, print -1.
    print(-1)