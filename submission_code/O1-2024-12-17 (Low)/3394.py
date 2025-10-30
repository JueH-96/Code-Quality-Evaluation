class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        We want an increasing sequence of length n, all containing the bits of x, with
        bitwise AND = x, and we wish to minimize the last element.

        Key idea:
        - Each number in the sequence must have all bits set that are set in x.
        - For bits that are 0 in x, we can choose to set them or not in each number,
          to obtain distinct, strictly increasing values.
        - A compact way to achieve exactly n distinct values is to map the binary
          representation of i (for i from 0 to n-1) onto the zero bits of x.
          Specifically, if bit j of i is 1, we set the j-th zero bit of x.

        Then, the last element (for i = n-1) is given by taking x and flipping on
        those zero-bit positions of x that correspond to the set bits of (n-1).

        Example check:
        1) n=3, x=4 (binary 0100) -> zero bits at positions [0,1,3,4,...]
           (n-1)=2 -> binary 10 means we set the 1st zero-bit of x (which is position 1)
           => last element = 0100 | 0010 = 0110 (binary) = 6

        2) n=2, x=7 (binary 0111) -> zero bits at positions [3,4,5,...]
           (n-1)=1 -> binary 1 means we set the 0th zero-bit (position 3)
           => last element = 0111 | 1000 = 1111 (binary) = 15
        """

        # Special case: if n=1, the only element is x itself, so the answer is x
        if n == 1:
            return x

        # Collect positions of bits that are 0 in x, in ascending order
        zero_positions = []
        # We'll go up to around 60 bits (enough for the problem constraints)
        for bit_pos in range(60):
            if not (x & (1 << bit_pos)):
                zero_positions.append(bit_pos)

        # We'll construct the final number by starting with x,
        # then setting certain zero-bits based on bits in (n-1).
        result = x
        mask = n - 1  # We'll use this to decide which zero bits to turn on

        # Turn on the corresponding zero-bits where (n-1) has a '1'.
        # Each set bit of mask sets the corresponding zero_positions element in 'result'.
        bit_idx = 0
        while mask > 0 and bit_idx < len(zero_positions):
            if (mask & 1) == 1:
                result |= (1 << zero_positions[bit_idx])
            bit_idx += 1
            mask >>= 1

        return result