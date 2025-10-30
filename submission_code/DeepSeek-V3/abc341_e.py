import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    S = data[idx]
    idx += 1
    # Initialize the flip array
    flip = [0] * (N + 2)
    for _ in range(Q):
        type_q = int(data[idx])
        L = int(data[idx+1])
        R = int(data[idx+2])
        idx += 3
        if type_q == 1:
            flip[L] ^= 1
            flip[R+1] ^= 1
        else:
            # Determine the current state of the characters in the range [L, R]
            # We need to find the number of flips for each character
            # The total flips for character at position i is the prefix sum of flip up to i
            # We can precompute the prefix sum
            # To handle multiple flips, we can compute the prefix sum on the fly
            # But since Q and N are up to 5e5, we need an efficient way
            # We can precompute the prefix sum of the flip array
            # Initialize the prefix sum
            # Since flip is a difference array, the prefix sum will give the total flips for each position
            # We can compute the prefix sum up to R and L-1
            # The total flips for position i is the prefix sum up to i
            # So for the range [L, R], we need to check if the parity of the flips for each character is consistent
            # But to check if the substring is good, we need to ensure that no two consecutive characters are the same
            # So, for each pair of consecutive characters in the range, their flips should be such that their original characters are different
            # Wait, perhaps a better approach is to precompute the parity of the flips for each character
            # Then, for the substring, we can determine the actual characters by XORing the original character with the parity of the flips
            # Then, we can check if all consecutive characters are different
            # So, first, compute the prefix sum of the flip array
            # Initialize the prefix sum array
            # Since flip is a difference array, the prefix sum will give the total flips for each position
            # We can compute the prefix sum up to R and L-1
            # The total flips for position i is the prefix sum up to i
            # So, for the range [L, R], we need to compute the prefix sum for each position in the range
            # But since Q is up to 5e5, we need an efficient way to compute the prefix sum for each query
            # We can precompute the prefix sum of the flip array
            # Initialize the prefix sum array
            prefix = [0] * (N + 2)
            for i in range(1, N+1):
                prefix[i] = prefix[i-1] ^ flip[i]
            # Now, for the range [L, R], the parity of the flips for each character is prefix[i]
            # So, the actual character at position i is S[i-1] ^ prefix[i]
            # We need to check if for all i in [L, R-1], S[i-1] ^ prefix[i] != S[i] ^ prefix[i+1]
            # So, for each i from L to R-1, check if (S[i-1] ^ prefix[i]) != (S[i] ^ prefix[i+1])
            # If all are true, then the substring is good
            is_good = True
            for i in range(L, R):
                current_char = int(S[i-1]) ^ prefix[i]
                next_char = int(S[i]) ^ prefix[i+1]
                if current_char == next_char:
                    is_good = False
                    break
            if is_good:
                print("Yes")
            else:
                print("No")

if __name__ == "__main__":
    main()