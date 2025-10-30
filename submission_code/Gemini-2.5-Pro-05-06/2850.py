class Solution:
  def longestString(self, x: int, y: int, z: int) -> int:
    # x: count of "AA" strings
    # y: count of "BB" strings
    # z: count of "AB" strings

    # Initialize total length
    length = 0

    # Contribution from "AB" strings:
    # All z "AB" strings can be used, for example, by forming (AB)^z.
    # Each "AB" string has length 2.
    length += 2 * z

    # Contribution from "AA" and "BB" strings:
    # We can form an alternating chain of "AA"s and "BB"s.
    # We can use min(x, y) "AA" strings and min(x, y) "BB" strings.
    # This forms min(x, y) pairs like ("AA", "BB") or ("BB", "AA").
    # Each pair contributes 2 blocks, total 2 * min(x,y) blocks.
    # Each block has length 2, so these contribute 2 * (2 * min(x, y)) = 4 * min(x, y) to length.
    length += 4 * min(x, y)

    # If x and y are not equal, one type of string ("AA" or "BB") is more numerous.
    # We can add one more block of this more numerous type to the alternating chain.
    # This additional block has length 2.
    if x != y:
      length += 2
      
    return length